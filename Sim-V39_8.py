import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import csv
import threading
import queue
import os
from datetime import datetime
from scipy.fft import rfft

try:
    import pandas as pd
    _PANDAS_OK = True
except ImportError:
    _PANDAS_OK = False


class WaveformSimulatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Waveform Predistortion Simulator - V39")
        self.root.geometry("1600x1000")

        self.target_mags = None
        self.target_phases = None
        self.freqs = None
        self.custom_loaded = False
        self.resonators = []
        self.last_time_data  = None
        self.last_V_node_req = None   # target spectrum (complex array)
        self.last_metrics    = {}     # power metrics dict from last _sim_core call
        self.last_rmo_lch_opt = None  # results from R_gen/L_choke optimizer
        self.V_mo_gen_base   = None
        self.V_hfi_gen_base  = None

        # ── Inductor database state (from Res_Opt) ──────────────────────
        self.inductor_db = None          # pandas DataFrame once loaded
        self.selected_inductors = []     # list of row dicts chosen by user

        # ── Cs-optimisation state ────────────────────────────────────────
        self._cs_optimized = False
        self._fast_eval = False          # can be toggled if needed
        self._plot_wins   = {}           # key → Toplevel
        self._plot_figs   = {}
        self._plot_canvas = {}

        # LED state for Cs optimisation progress indicator
        self._led_on  = False
        self._led_job = None
        self._led_canvas = None
        self._led       = None
        self._opt_btn   = None

        self.create_menu()
        self.create_input_panel()
        self.create_plot_panel()

        default_res = [(1.25e02, 2.55e-13, 8.20e-07, 9.80e-04, 2.40e+00, 4.28564e-11, 5.00e-13)]
        self.build_resonator_tabs(default_res)

        self.update_fres()
        self._update_hfi_harms_label()
        self.calculate_and_plot()
    # ═══════════════════════════════════════════════════════════════════════

    def create_menu(self):
        menubar = tk.Menu(self.root)

        # ── File menu ────────────────────────────────────────────────────
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Load Waveform (CSV)",          command=self.load_csv)
        file_menu.add_command(label="Save Predistorted (CSV)",       command=self.save_csv)
        file_menu.add_command(label="Clear Loaded CSV",              command=self.clear_csv)
        file_menu.add_separator()
        file_menu.add_command(label="Load Resonators (*.res)",       command=self.load_resonators)
        file_menu.add_command(label="Save Resonators (*.res)",       command=self.save_resonators)
        file_menu.add_separator()
        file_menu.add_command(label="Open Inductor Database (CSV)",  command=self._menu_open_inductor_db)
        file_menu.add_separator()
        file_menu.add_command(label="Save Generator Spectra (CSV)",  command=self.save_spectra)
        file_menu.add_separator()
        file_menu.add_command(label="Save Time Dependences (Plot Image)", command=self.save_time_plot)
        file_menu.add_command(label="Save Time Dependences (CSV Data)",   command=self.save_time_data)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        # ── Tuning menu ──────────────────────────────────────────────────
        tuning_menu = tk.Menu(menubar, tearoff=0)

        # "Choose Inductors" — will be populated once a DB is loaded
        self.inductor_submenu = tk.Menu(tuning_menu, tearoff=0)
        self.inductor_submenu.add_command(label="(Load inductor database first)",
                                          state="disabled")
        tuning_menu.add_cascade(label="Choose Inductors", menu=self.inductor_submenu)

        tuning_menu.add_separator()
        tuning_menu.add_command(label="Optimize R_gen / L_choke for max FOM/N",
                                command=self._run_rmo_lch_optimizer)

        menubar.add_cascade(label="Tuning", menu=tuning_menu)

        self.root.config(menu=menubar)

    # ═══════════════════════════════════════════════════════════════════════
    # INPUT PANEL
    # ═══════════════════════════════════════════════════════════════════════

    def create_input_panel(self):
        self.control_canvas = tk.Canvas(self.root, width=520)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.control_canvas.yview)
        self.scrollable_frame = ttk.Frame(self.control_canvas, padding="10")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all")))
        self.control_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.control_canvas.configure(yscrollcommand=scrollbar.set)
        self.control_canvas.pack(side=tk.LEFT, fill=tk.Y, expand=False)
        scrollbar.pack(side=tk.LEFT, fill=tk.Y)

        def add_entry(parent, label_text, default_val, row, col=0,
                      trace_func=None, width=10):
            ttk.Label(parent, text=label_text).grid(
                row=row, column=col, sticky=tk.W, pady=2, padx=(0, 5))
            var = tk.StringVar(value=str(default_val))
            if trace_func:
                var.trace_add("write", trace_func)
            entry = ttk.Entry(parent, textvariable=var, width=width)
            entry.grid(row=row, column=col + 1, sticky=tk.EW, pady=2, padx=(0, 10))
            return var
        self.add_entry = add_entry

        # --- 1) Base Parameters ---
        lf_base = ttk.LabelFrame(self.scrollable_frame, text="Base Parameters", padding="5")
        lf_base.grid(row=0, column=0, sticky=tk.EW, pady=5)
        self.v_f = add_entry(lf_base, "Freq (MHz):", "33",  row=0, col=0)
        self.v_n = add_entry(lf_base, "N Harmonics:", "10", row=0, col=2)

        # --- 2) Trapezoid Shape (2 lines) ---
        lf_trap = ttk.LabelFrame(self.scrollable_frame, text="Trapezoid Shape", padding="5")
        lf_trap.grid(row=1, column=0, sticky=tk.EW, pady=5)
        self.v_delay = add_entry(lf_trap, "Delay (x T):",   "0.1",   row=0, col=0)
        self.v_ramp  = add_entry(lf_trap, "Ramp (x T):",    "0.032", row=0, col=2)
        self.v_up    = add_entry(lf_trap, "Up Time (x T):", "0.2",   row=0, col=4)
        self.v_vmin  = add_entry(lf_trap, "V_min (V):",     "0",     row=1, col=0)
        self.v_vmax  = add_entry(lf_trap, "V_max (V):",     "1",     row=1, col=2)

        # --- 3) Circuit Components (3 lines) + MO Harmonics ---
        lf_circ = ttk.LabelFrame(self.scrollable_frame, text="Circuit Components", padding="5")
        lf_circ.grid(row=2, column=0, sticky=tk.EW, pady=5)
        self.v_rload = add_entry(lf_circ, "R_load (Ω):",    "1.0",   row=0, col=0)
        self.v_cload = add_entry(lf_circ, "C_load (pF):",   "50.0",  row=0, col=2,
                                 trace_func=self.update_fres)
        self.v_ltl   = add_entry(lf_circ, "L_TL (nH):",     "0.1",   row=0, col=4)
        self.v_rmo   = add_entry(lf_circ, "R_MO (Ω):",      "50.0",  row=1, col=0)
        self.v_rgnd  = add_entry(lf_circ, "R_toGND (Ω):",   "1e10",  row=1, col=2)
        self.v_lch   = add_entry(lf_circ, "L_choke (nH):",  "100.0", row=1, col=4)
        self.v_rhfi  = add_entry(lf_circ, "R_HFI (Ω):",     "50.0",  row=2, col=0)
        self.v_chi   = add_entry(lf_circ, "C_hi (pF):",     "1.0",   row=2, col=2,
                                 trace_func=self.update_fres)
        ttk.Label(lf_circ, text="MO Harm:").grid(
            row=2, column=4, sticky=tk.W, pady=2, padx=(4, 2))
        self.v_mo_harms = tk.StringVar(value="0,1")
        ttk.Entry(lf_circ, textvariable=self.v_mo_harms, width=6).grid(
            row=2, column=5, sticky=tk.EW, pady=2, padx=(0, 4))

        # HFI harmonics computed label
        hfi_row = ttk.Frame(lf_circ)
        hfi_row.grid(row=3, column=0, columnspan=6, sticky=tk.EW, pady=(0, 3))
        ttk.Label(hfi_row, text="HFI covers:").pack(side=tk.LEFT, padx=(0, 5))
        self.lbl_hfi_harms = ttk.Label(hfi_row, text="--",
                                        foreground="darkblue", font=('Arial', 9))
        self.lbl_hfi_harms.pack(side=tk.LEFT)
        self.v_mo_harms.trace_add("write", self._update_hfi_harms_label)
        self.v_n.trace_add("write",         self._update_hfi_harms_label)
        self.v_chi.trace_add("write",        self._update_hfi_harms_label)
        self.v_rgnd.trace_add("write",       lambda *a: self._get_rgnd())

        # --- Resonators ---
        lf_res = ttk.LabelFrame(self.scrollable_frame,
                                 text="Resonators (Dynamic Complex Model)", padding="5")
        lf_res.grid(row=3, column=0, sticky=tk.EW, pady=5)

        switch_frame = ttk.Frame(lf_res)
        switch_frame.pack(side=tk.TOP, anchor=tk.W, pady=(0, 5))
        ttk.Label(switch_frame, text="GLOBAL ENABLE:",
                  font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=(0, 8))

        self.v_global_res_enable = tk.BooleanVar(value=True)
        self.sw_canvas = tk.Canvas(switch_frame, width=40, height=20, highlightthickness=0)
        self.sw_canvas.pack(side=tk.LEFT)
        self.sw_canvas.bind("<Button-1>",
            lambda e: self.v_global_res_enable.set(not self.v_global_res_enable.get()))

        # LED indicator for Cs optimisation
        led_frame = ttk.Frame(switch_frame)
        led_frame.pack(side=tk.LEFT, padx=(20, 0))
        ttk.Label(led_frame, text="Cs opt:", font=('Arial', 8)).pack(side=tk.LEFT)
        self._led_canvas = tk.Canvas(led_frame, width=14, height=14, highlightthickness=0)
        self._led_canvas.pack(side=tk.LEFT, padx=(3, 0))
        self._led = self._led_canvas.create_oval(1, 1, 13, 13, fill="#cccccc", outline="")

        def draw_switch(*args):
            self.sw_canvas.delete("all")
            if self.v_global_res_enable.get():
                self.sw_canvas.create_oval(20, 0, 40, 20, fill="#4CAF50", outline="")
                self.sw_canvas.create_oval(0,  0, 20, 20, fill="#4CAF50", outline="")
                self.sw_canvas.create_rectangle(10, 0, 30, 20, fill="#4CAF50", outline="")
                self.sw_canvas.create_oval(22, 2, 38, 18, fill="white",   outline="")
            else:
                self.sw_canvas.create_oval(0,  0, 20, 20, fill="#aaa", outline="")
                self.sw_canvas.create_oval(20, 0, 40, 20, fill="#aaa", outline="")
                self.sw_canvas.create_rectangle(10, 0, 30, 20, fill="#aaa", outline="")
                self.sw_canvas.create_oval(2,  2, 18, 18, fill="white", outline="")

        self.v_global_res_enable.trace_add("write", draw_switch)
        draw_switch()

        self.notebook_res = ttk.Notebook(lf_res)
        self.notebook_res.pack(fill=tk.BOTH, expand=True)

        # --- AC Frequency Sweep ---
        lf_sweep = ttk.LabelFrame(self.scrollable_frame, text="AC Frequency Sweep", padding="5")
        lf_sweep.grid(row=4, column=0, sticky=tk.EW, pady=5)
        self.v_f_start = add_entry(lf_sweep, "Start (MHz):", "1.0",    row=0, col=0, width=8)
        self.v_f_stop  = add_entry(lf_sweep, "Stop (MHz):",  "1000.0", row=0, col=2, width=8)
        self.v_f_pts   = add_entry(lf_sweep, "Points:",      "1000",   row=0, col=4, width=7)

        # --- Power Metrics & FOM ---
        lf_metrics = ttk.LabelFrame(self.scrollable_frame,
                                     text="Power Metrics & FOM", padding="5")
        lf_metrics.grid(row=5, column=0, sticky=tk.EW, pady=5)

        self.lbl_p_in_total = ttk.Label(
            lf_metrics, text="Total DC Power Drawn: -- W",
            font=('Arial', 10, 'bold'))
        self.lbl_p_in_total.grid(row=0, column=0, sticky=tk.W, pady=2)

        self.lbl_p_baseline = ttk.Label(
            lf_metrics, text="  Baseline (no resonators): -- W",
            font=('Arial', 9), foreground="gray")
        self.lbl_p_baseline.grid(row=1, column=0, sticky=tk.W, pady=0)

        self.lbl_p_load = ttk.Label(
            lf_metrics,
            text="R_Load loss (unavoidable): -- W | R_MO loss: -- W | R_HFI loss: -- W",
            font=('Arial', 9))
        self.lbl_p_load.grid(row=2, column=0, sticky=tk.W, pady=2)

        self.lbl_p_res = ttk.Label(
            lf_metrics,
            text="Resonator heat (R1+R_DC+R_var): -- W",
            font=('Arial', 9))
        self.lbl_p_res.grid(row=3, column=0, sticky=tk.W, pady=2)

        self.lbl_p_waste = ttk.Label(
            lf_metrics,
            text="Cap. discharge returned to gen (not recycled): -- W",
            font=('Arial', 9))
        self.lbl_p_waste.grid(row=4, column=0, sticky=tk.W, pady=2)

        ttk.Separator(lf_metrics, orient='horizontal').grid(
            row=5, column=0, sticky=tk.EW, pady=5)

        self.lbl_fom = ttk.Label(
            lf_metrics, text="Recycl. FOM: --",
            font=('Arial', 11, 'bold'), foreground="purple")
        self.lbl_fom.grid(row=6, column=0, sticky=tk.W, pady=(2, 1))

        self.lbl_eta_del = ttk.Label(
            lf_metrics, text="Delivery eff. η_del (E_load/E_drawn): --   |   R_MO eff. η_MO: --",
            font=('Arial', 9), foreground="darkgreen")
        self.lbl_eta_del.grid(row=7, column=0, sticky=tk.W, pady=(0, 2))

        ttk.Separator(lf_metrics, orient='horizontal').grid(
            row=8, column=0, sticky=tk.EW, pady=4)

        ttk.Label(lf_metrics,
                  text="Energy per cycle  (pJ):",
                  font=('Arial', 9, 'bold')).grid(
            row=9, column=0, sticky=tk.W, pady=(2, 1))

        self.lbl_e_in = ttk.Label(lf_metrics,
            text="  E_drawn (from MO+HFI):               -- pJ",
            font=('Arial', 9))
        self.lbl_e_in.grid(row=10, column=0, sticky=tk.W)

        self.lbl_e_load = ttk.Label(lf_metrics,
            text="  E_Rload (unavoidable I²R in R_Load):  -- pJ",
            font=('Arial', 9))
        self.lbl_e_load.grid(row=11, column=0, sticky=tk.W)

        self.lbl_e_rmo = ttk.Label(lf_metrics,
            text="  E_R_MO  (I²R loss, MO branch):       -- pJ",
            font=('Arial', 9))
        self.lbl_e_rmo.grid(row=12, column=0, sticky=tk.W)

        self.lbl_e_res = ttk.Label(lf_metrics,
            text="  E_res   (loss in resonators):        -- pJ",
            font=('Arial', 9))
        self.lbl_e_res.grid(row=13, column=0, sticky=tk.W)

        self.lbl_e_dump = ttk.Label(lf_metrics,
            text="  E_ret   (cap discharge, not recycled): -- pJ",
            font=('Arial', 9))
        self.lbl_e_dump.grid(row=14, column=0, sticky=tk.W)

        ttk.Separator(lf_metrics, orient='horizontal').grid(
            row=15, column=0, sticky=tk.EW, pady=3)

        self.lbl_e_cv2 = ttk.Label(lf_metrics,
            text="  ½CV²_max (adiabatic no-recycle limit): -- pJ",
            font=('Arial', 9), foreground="navy")
        self.lbl_e_cv2.grid(row=16, column=0, sticky=tk.W)

        self.lbl_e_cv2full = ttk.Label(lf_metrics,
            text="  CV²_max  (fast-switch limit):          -- pJ",
            font=('Arial', 9), foreground="firebrick")
        self.lbl_e_cv2full.grid(row=17, column=0, sticky=tk.W, pady=(0, 2))

    # ═══════════════════════════════════════════════════════════════════════
    # PLOT PANEL
    # ═══════════════════════════════════════════════════════════════════════

    def create_plot_panel(self):
        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.plot_notebook = ttk.Notebook(self.plot_frame)
        self.plot_notebook.pack(fill=tk.BOTH, expand=True)

        # ── Time Domain tab ──────────────────────────────────────────────
        self.tab_time = ttk.Frame(self.plot_notebook)
        self.plot_notebook.add(self.tab_time, text="Time Domain")

        btn_time_frame = ttk.Frame(self.tab_time)
        btn_time_frame.pack(side=tk.TOP, fill=tk.X, pady=(6, 2))
        btn_row = ttk.Frame(btn_time_frame)
        btn_row.pack(anchor=tk.CENTER)
        ttk.Button(btn_row, text="▶  Run Simulation",
                   command=self.calculate_and_plot,
                   style="Accent.TButton").pack(side=tk.LEFT, ipadx=20, ipady=4, padx=(0, 16))
        self.v_predist_off = tk.BooleanVar(value=False)
        ttk.Checkbutton(btn_row, text="Show undistorted (predistortion OFF)",
                        variable=self.v_predist_off).pack(side=tk.LEFT, padx=(0, 4))

        self.fig_time = plt.Figure(figsize=(11, 8), dpi=100)
        self.gs_time  = self.fig_time.add_gridspec(2, 2)
        self.ax_time_load  = self.fig_time.add_subplot(self.gs_time[0, 0])
        self.ax_time_gen   = self.fig_time.add_subplot(self.gs_time[0, 1])
        self.ax_time_gen2  = self.ax_time_gen.twinx()
        self.ax_energy     = self.fig_time.add_subplot(self.gs_time[1, 0])
        self.ax_current    = self.fig_time.add_subplot(self.gs_time[1, 1])
        self.fig_time.tight_layout(pad=3.0)
        self.canvas_time = FigureCanvasTkAgg(self.fig_time, master=self.tab_time)
        self.canvas_time.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ── Frequency Domain tab ─────────────────────────────────────────
        self.tab_freq = ttk.Frame(self.plot_notebook)
        self.plot_notebook.add(self.tab_freq, text="Frequency Domain")

        btn_freq_frame = ttk.Frame(self.tab_freq)
        btn_freq_frame.pack(side=tk.TOP, fill=tk.X, pady=(6, 2))
        btn_freq_row = ttk.Frame(btn_freq_frame)
        btn_freq_row.pack(anchor=tk.CENTER)
        ttk.Button(btn_freq_row, text="⟳  Run Frequency Sweep",
                   command=self.run_ac_sweep).pack(side=tk.LEFT, ipadx=20, ipady=4, padx=(0,10))
        ttk.Button(btn_freq_row, text="📊  Calculate Spectra",
                   command=self.calculate_spectra).pack(side=tk.LEFT, ipadx=20, ipady=4)

        self.fig_freq = plt.Figure(figsize=(11, 8), dpi=100)
        self.gs_freq  = self.fig_freq.add_gridspec(2, 2)
        self.ax_freq_mag   = self.fig_freq.add_subplot(self.gs_freq[0, 0])
        self.ax_freq_phase = self.fig_freq.add_subplot(self.gs_freq[0, 1])
        self.ax_vload_f    = self.fig_freq.add_subplot(self.gs_freq[1, 0])
        self.ax_iratio_f   = self.fig_freq.add_subplot(self.gs_freq[1, 1])
        self.fig_freq.tight_layout(pad=3.0)
        self.canvas_freq = FigureCanvasTkAgg(self.fig_freq, master=self.tab_freq)
        self.canvas_freq.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ── Cs Tuning tab ────────────────────────────────────────────────
        self.tab_cs = ttk.Frame(self.plot_notebook)
        self.plot_notebook.add(self.tab_cs, text="Cs Tuning")

        btn_cs_frame = ttk.Frame(self.tab_cs)
        btn_cs_frame.pack(side=tk.TOP, fill=tk.X, pady=(6, 2))
        ttk.Button(btn_cs_frame, text="▶  Run Cs Optimization",
                   command=self._run_cs_optimization).pack(
            side=tk.TOP, anchor=tk.CENTER, ipadx=20, ipady=4)

        self.fig_cs = plt.Figure(figsize=(11, 8), dpi=100)
        self.canvas_cs = FigureCanvasTkAgg(self.fig_cs, master=self.tab_cs)
        tb_cs = NavigationToolbar2Tk(self.canvas_cs, self.tab_cs)
        tb_cs.update()
        tb_cs.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas_cs.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._cs_status_var = tk.StringVar(value="Press 'Run Cs Optimization' to start.")
        ttk.Label(self.tab_cs, textvariable=self._cs_status_var,
                  relief="sunken", anchor="w").pack(side=tk.BOTTOM, fill=tk.X)

    # ═══════════════════════════════════════════════════════════════════════
    # INDUCTOR DATABASE  (ported from Res_Opt_v2_01)
    # ═══════════════════════════════════════════════════════════════════════

    def _menu_open_inductor_db(self):
        if not _PANDAS_OK:
            messagebox.showerror("Missing dependency",
                                 "pandas is required to load an inductor database.\n"
                                 "Install it with:  pip install pandas")
            return
        filepath = filedialog.askopenfilename(
            title="Open Inductor Database (CSV)",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
        if not filepath:
            return
        try:
            df = pd.read_csv(filepath)
            required = {"Part_Number", "R1_Ohm", "R2_Ohm", "C_pF",
                        "L_nH", "k", "Upper_limit_MHz"}
            if not required.issubset(set(df.columns)):
                messagebox.showerror(
                    "Invalid File",
                    f"CSV must contain columns:\n{', '.join(sorted(required))}")
                return
            self.inductor_db = df
            messagebox.showinfo(
                "Database Loaded",
                f"Loaded {len(df)} inductors from:\n{filepath}")
            self._rebuild_inductor_submenu()
        except Exception as e:
            messagebox.showerror("Error loading database", str(e))

    def _rebuild_inductor_submenu(self):
        """Repopulate Tuning → Choose Inductors after a database is loaded."""
        self.inductor_submenu.delete(0, "end")
        if self.inductor_db is None or self.inductor_db.empty:
            self.inductor_submenu.add_command(
                label="(No database loaded)", state="disabled")
            return
        self.inductor_submenu.add_command(
            label="Select Inductors…", command=self._open_inductor_selector)

    # -----------------------------------------------------------------------
    # Inductor Selector Dialog (checkbox-based, order-aware — from Res_Opt)
    # -----------------------------------------------------------------------

    def _open_inductor_selector(self):
        if self.inductor_db is None:
            messagebox.showinfo("No Database",
                                "Please load an inductor database first.")
            return

        dlg = tk.Toplevel(self.root)
        dlg.title("Choose Inductors — click to select (order matters)")
        dlg.grab_set()
        dlg.resizable(True, True)
        dlg.geometry("820x540")

        cols = list(self.inductor_db.columns)
        rows = self.inductor_db.to_dict("records")

        selection_order = []
        summary_var = tk.StringVar(value="0 inductors selected")

        COL_BG_EVEN = "#f5f5f5"
        COL_BG_ODD  = "#ffffff"
        COL_BG_SEL  = "#d0e8ff"
        BADGE_BG    = "#2a7ae2"
        BADGE_EMPTY = "#cccccc"

        num_labels  = []
        row_widgets = []

        def refresh():
            for i in range(len(rows)):
                pos = selection_order.index(i) + 1 if i in selection_order else 0
                if pos:
                    num_labels[i].config(text=str(pos), bg=BADGE_BG, fg="white",
                                         font=("Helvetica", 9, "bold"))
                else:
                    num_labels[i].config(text="", bg=BADGE_EMPTY, fg=BADGE_EMPTY)
                base = COL_BG_EVEN if i % 2 == 0 else COL_BG_ODD
                bg   = COL_BG_SEL if i in selection_order else base
                for w in row_widgets[i]:
                    try:
                        w.config(bg=bg)
                    except tk.TclError:
                        pass
            n     = len(selection_order)
            names = [rows[i]["Part_Number"] for i in selection_order]
            summary_var.set(
                f"{n} selected: {', '.join(names)}" if n else "0 inductors selected")

        # scrollable table
        outer = ttk.Frame(dlg)
        outer.pack(fill="both", expand=True, padx=6, pady=(4, 0))
        cv  = tk.Canvas(outer, borderwidth=0, highlightthickness=0)
        vsb = ttk.Scrollbar(outer, orient="vertical", command=cv.yview)
        cv.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        cv.pack(side="left", fill="both", expand=True)
        inner  = ttk.Frame(cv)
        win_id = cv.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: cv.configure(scrollregion=cv.bbox("all")))
        cv.bind("<Configure>",
                lambda e: cv.itemconfig(win_id, width=e.width))

        def _mw(e):
            cv.yview_scroll(int(-1 * (e.delta / 120)), "units")
        cv.bind_all("<MouseWheel>", _mw)

        # header
        col_widths = {"Part_Number": 14, "R1_Ohm": 7, "R2_Ohm": 7,
                      "C_pF": 6, "L_nH": 8, "k": 8, "Upper_limit_MHz": 10}
        HDR = ("Helvetica", 9, "bold")
        tk.Label(inner, text="#", width=3, font=HDR,
                 relief="ridge", anchor="center",
                 bg="#dde8f5").grid(row=0, column=0, padx=1, pady=1, sticky="nsew")
        tk.Label(inner, text="Select", width=6, font=HDR,
                 relief="ridge", anchor="center",
                 bg="#dde8f5").grid(row=0, column=1, padx=1, pady=1, sticky="nsew")
        for ci, c in enumerate(cols):
            tk.Label(inner, text=c, width=col_widths.get(c, 8),
                     font=HDR, relief="ridge", anchor="center",
                     bg="#dde8f5").grid(row=0, column=ci + 2,
                                        padx=1, pady=1, sticky="nsew")

        # data rows
        cb_vars = [tk.BooleanVar(value=False) for _ in rows]

        def _cmd(i):
            if cb_vars[i].get():
                if i not in selection_order:
                    selection_order.append(i)
            else:
                if i in selection_order:
                    selection_order.remove(i)
            refresh()

        for ri, row in enumerate(rows):
            base_bg = COL_BG_EVEN if ri % 2 == 0 else COL_BG_ODD
            wlist   = []

            nl = tk.Label(inner, text="", width=3, anchor="center",
                          bg=BADGE_EMPTY, fg=BADGE_EMPTY,
                          font=("Helvetica", 9, "bold"), relief="flat")
            nl.grid(row=ri + 1, column=0, padx=1, pady=1, sticky="nsew")
            num_labels.append(nl)
            wlist.append(nl)

            cb = tk.Checkbutton(inner, variable=cb_vars[ri],
                                bg=base_bg, activebackground=base_bg)
            cb.config(command=lambda i=ri: _cmd(i))
            cb.grid(row=ri + 1, column=1, padx=4, pady=1, sticky="nsew")
            wlist.append(cb)

            for ci, c in enumerate(cols):
                val = row[c]
                txt = f"{val:.4g}" if isinstance(val, float) else str(val)
                lbl = tk.Label(inner, text=txt, width=col_widths.get(c, 8),
                               anchor="center", bg=base_bg,
                               font=("Helvetica", 9))
                lbl.grid(row=ri + 1, column=ci + 2, padx=1, pady=1, sticky="nsew")

                def _row_click(e, i=ri):
                    cb_vars[i].set(not cb_vars[i].get())
                    _cmd(i)
                lbl.bind("<Button-1>", _row_click)
                wlist.append(lbl)

            row_widgets.append(wlist)

        # summary + buttons
        ttk.Label(dlg, textvariable=summary_var,
                  foreground="darkblue", font=("Helvetica", 9, "bold"),
                  padding=4).pack(anchor="w")

        btn_f = ttk.Frame(dlg, padding=6)
        btn_f.pack(fill="x")

        def confirm():
            if not selection_order:
                messagebox.showwarning("Nothing selected",
                                       "Please check at least one inductor.",
                                       parent=dlg)
                return
            self.selected_inductors = [rows[i] for i in selection_order]
            self._apply_inductors_to_resonator_tabs()
            dlg.destroy()

        ttk.Button(btn_f, text="Add to Resonator Tabs",
                   command=confirm).pack(side="left", padx=5)
        ttk.Button(btn_f, text="Clear All",
                   command=lambda: (selection_order.clear(),
                                    [v.set(False) for v in cb_vars],
                                    refresh())).pack(side="left", padx=5)
        ttk.Button(btn_f, text="Cancel",
                   command=dlg.destroy).pack(side="left", padx=5)

        dlg.protocol("WM_DELETE_WINDOW",
                     lambda: (cv.unbind_all("<MouseWheel>"), dlg.destroy()))

    def _apply_inductors_to_resonator_tabs(self):
        """
        Map selected inductors from the DB into resonator tabs.
          R1_Ohm -> R1,  R2_Ohm -> R_DC,  C_pF -> C1 (F),  L_nH -> L (H),  k -> k
        Prompts user to keep or reset existing Cs / Cp values.
        """
        if not self.selected_inductors:
            return

        NEW_DEFAULT_Cs = 1.0e-12
        NEW_DEFAULT_Cp = 0.5e-12
        n_new = len(self.selected_inductors)

        # Harvest existing Cs/Cp from current tabs
        existing_cs_cp = []
        for res in self.resonators:
            try:
                cs = float(res['cs'].get())
                cp = float(res['cp'].get())
                existing_cs_cp.append((cs, cp))
            except ValueError:
                existing_cs_cp.append((NEW_DEFAULT_Cs, NEW_DEFAULT_Cp))

        keep_cs = False
        if existing_cs_cp:
            n_old = len(existing_cs_cp)
            if n_new != n_old:
                msg = (f"Number of resonators changed ({n_old} → {n_new}).\n\n"
                       f"Keep existing Cs/Cp for the first {min(n_old, n_new)} rows?\n"
                       f"New rows will use defaults: "
                       f"Cs={NEW_DEFAULT_Cs*1e12:.4g} pF, Cp={NEW_DEFAULT_Cp*1e12:.4g} pF.")
            else:
                msg = (f"Keep existing Cs/Cp values for all {n_new} rows?\n\n"
                       f"(No = reset to defaults: "
                       f"Cs={NEW_DEFAULT_Cs*1e12:.4g} pF, Cp={NEW_DEFAULT_Cp*1e12:.4g} pF)")
            keep_cs = messagebox.askyesno("Cs / Cp Assignment", msg)

        # Build res_data list for build_resonator_tabs
        res_data = []
        for i, d in enumerate(self.selected_inductors):
            R1   = float(d["R1_Ohm"])
            C1   = float(d["C_pF"]) * 1e-12
            L    = float(d["L_nH"]) * 1e-9
            k    = float(d["k"])
            R_DC = float(d["R2_Ohm"])
            if keep_cs and i < len(existing_cs_cp):
                Cs, Cp = existing_cs_cp[i]
            else:
                Cs, Cp = NEW_DEFAULT_Cs, NEW_DEFAULT_Cp
            res_data.append((R1, C1, L, k, R_DC, Cs, Cp))

        self.build_resonator_tabs(res_data)
        self.update_fres()
        self.calculate_and_plot()
        names = [d["Part_Number"] for d in self.selected_inductors]
        messagebox.showinfo("Inductors Applied",
                            f"Loaded {len(names)} inductor(s):\n{', '.join(names)}")

    # ═══════════════════════════════════════════════════════════════════════
    # CS OPTIMISATION  (protocol from Res_Opt_v1_34)
    #
    # Resonator k (0-indexed) is tuned so its anti-resonance falls at:
    #   f_target = (k+1) * f_base
    #
    # The anti-resonance is found by maximising |Iload/Iser| at f_target,
    # where:
    #   Z_load   = Rload + jwL_TL + 1/(jwCload)
    #   Z_struct = Z_res_total || Z_load
    #   ratio    = |Z_struct / Z_load|
    #
    # This is identical to Res_Opt_v1_34._current_ratio_at / _cs_for_max_ratio.
    # Cload is passed in Farads; L_TL maps to Lload; R_mo maps to Rser.
    # C_hi is NOT part of Z_load (consistent with user's anti-res formula).
    # ═══════════════════════════════════════════════════════════════════════

    def _single_res_impedance(self, f, params_row):
        """
        params_row: [R1, C1(F), L(H), k, R_DC, Cs(F), Cp(F), ...]
        Identical to Res_Opt_v1_34.calculate_single_resonator_impedance.
        Cs < 1e-16 F → open circuit (returns inf).
        """
        R1, C1, L, k, R_DC, Cs, Cp = params_row[:7]
        if f < 1e-6 or Cs < 1e-16:
            return np.inf
        omega = 2.0 * np.pi * f
        Z_A   = R1 + 1.0 / (1j * omega * C1)
        R_se  = k * np.sqrt(f)
        Z_B   = (R_se + R_DC) + 1j * omega * L
        Z_AB  = 1.0 / (1.0/Z_A + 1.0/Z_B)
        Z_ser = Z_AB + 1.0 / (1j * omega * Cs)
        return 1.0 / (1.0/Z_ser + 1j * omega * Cp)

    def _current_ratio_at(self, f, params, Rload, Cload_F, Lload, R_HFI=0.0, C_hi_val=0.0):
        """
        |I_load / I_choke| at frequency f — identical quantity to what run_ac_sweep plots.

        I_load  = V_node * Y_load_base   (current into the R_load+L_TL+C_load branch only)
        I_choke = current into the whole parallel combination from the series choke
                = V_node * (Y_load_base + Y_hfi + Y_res)
        ratio   = |Y_load_base| / |Y_load_base + Y_hfi + Y_res|

        C_hi (Y_hfi) is a SEPARATE parallel branch, NOT folded into Z_load.
        Without any resonator (Y_res=0) the baseline is |Y_load_base|/|Y_load_base+Y_hfi|,
        which is < 1 when C_hi is large. The resonator is useful when its anti-resonance
        raises the ratio above that baseline.

        params   : numpy array (N, 8+)
        Cload_F  : C_load in Farads (load branch only, not including C_hi)
        Lload    : L_TL in Henries
        """
        w = 2.0 * np.pi * f
        if w == 0.0:
            return 0.0
        Y_res = sum(
            1.0 / self._single_res_impedance(f, p)
            for p in params if p[5] >= 1e-16
            and np.isfinite(abs(self._single_res_impedance(f, p)))
        )
        Z_res_total = 1.0 / Y_res if abs(Y_res) > 1e-30 else np.inf

        Z_load_main = Rload + 1j * w * Lload + 1.0 / (1j * w * Cload_F)
        Y_load_base = 1.0 / Z_load_main

        Y_hfi = (1.0 / (R_HFI + 1.0 / (1j * w * C_hi_val))
                 if C_hi_val > 1e-15 else 0.0)

        Y_res_val = (1.0 / Z_res_total
                     if np.isfinite(abs(Z_res_total)) else 0.0)

        denom = Y_load_base + Y_hfi + Y_res_val
        return abs(Y_load_base) / abs(denom) if abs(denom) > 1e-30 else 0.0

    def _cs_for_max_ratio(self, params, row_idx, f_target, Rload, Cload_F, Lload,
                          R_HFI=0.0, C_hi_val=0.0):
        """
        Scan Cs for resonator row_idx to maximise |Iload/Iser| at f_target.

        cs_max is set dynamically: the optimal Cs for anti-resonance at f_target is
            Cs_opt = c_eff_t * C_par / (C_par - c_eff_t)
        where c_eff_t = 1/(L·ω²) and C_par = Cp + Cload + C_hi.
        If C_par ≤ c_eff_t, no anti-resonance is possible at f_target with this L.
        We search up to 20× Cs_opt to cover the full useful range.

        Returns optimal Cs in Farads, or 1e-17 if no useful value found.
        """
        p_work = params.copy()

        def neg_ratio(log_cs):
            p_work[row_idx, 5] = 10.0 ** log_cs
            return -self._current_ratio_at(f_target, p_work, Rload, Cload_F, Lload,
                                           R_HFI, C_hi_val)

        L = params[row_idx, 2]
        Cp_val = params[row_idx, 6]

        # cs_series: Cs value that puts series resonance of L+Cs AT f_target
        cs_series_res = (1.0 / (L * (2.0 * np.pi * f_target) ** 2)
                         if L > 0 else 1e-12)
        cs_min = max(0.001e-12, cs_series_res * 0.1)

        # Dynamic cs_max using the correct system anti-resonance condition.
        # Cp, Cload, and Chi are all in parallel at the node, so the total
        # shunt capacitance the motional branch sees is C_par = Cp + Cload + Chi.
        # System anti-resonance at f_target requires C_par > 1/(L·ω²) = cs_series_res,
        # and the optimal Cs is: Cs_opt = cs_series_res * C_par / (C_par - cs_series_res)
        Cp_val = params[row_idx, 6]
        C_par_total = Cp_val + Cload_F + C_hi_val
        if L > 0 and C_par_total > cs_series_res:
            cs_opt_analytic = cs_series_res * C_par_total / (C_par_total - cs_series_res)
            cs_max = max(1000e-12, cs_opt_analytic * 20.0)
        else:
            cs_max = 1000e-12  # C_par too small — no anti-res possible, search anyway

        n_coarse = 60  if self._fast_eval else 500
        n_fine   = 30  if self._fast_eval else 200
        log_grid  = np.linspace(np.log10(cs_min), np.log10(cs_max), n_coarse)
        neg_vals  = np.array([neg_ratio(lc) for lc in log_grid])
        best_idx  = int(np.argmin(neg_vals))
        lo        = log_grid[max(0, best_idx - 2)]
        hi        = log_grid[min(len(log_grid) - 1, best_idx + 2)]
        fine_grid = np.linspace(lo, hi, n_fine)
        fine_vals = np.array([neg_ratio(lc) for lc in fine_grid])
        best_log  = fine_grid[int(np.argmin(fine_vals))]
        best_cs   = 10.0 ** best_log

        if best_cs < 1e-16:
            return 1e-17
        # Reject if resonator doesn't improve the ratio beyond the no-resonator baseline.
        w_t = 2.0 * np.pi * f_target
        Z_load_bl = Rload + 1j * w_t * Lload + 1.0 / (1j * w_t * Cload_F)
        Y_load_bl  = 1.0 / Z_load_bl
        Y_hfi_bl   = (1.0 / (R_HFI + 1.0 / (1j * w_t * C_hi_val))
                      if C_hi_val > 1e-15 else 0.0)
        denom_bl   = Y_load_bl + Y_hfi_bl
        baseline   = (abs(Y_load_bl) / abs(denom_bl)
                      if abs(denom_bl) > 1e-30 else 1.0)
        if -min(fine_vals) <= baseline:
            return 1e-17
        # Sub-harmonic reject: verify this resonator's own peak is near f_target.
        # Use only this resonator in isolation — the combined system peak shifts
        # when multiple resonators interact, causing false rejects.
        harm_num  = params[row_idx, 7]
        f0_est    = f_target / harm_num
        f_scan    = np.linspace(f0_est * 0.5, f_target * 1.5, 120)
        p_single  = p_work[row_idx:row_idx+1].copy()   # this resonator only
        ratio_scan = np.array([
            self._current_ratio_at(fs, p_single, Rload, Cload_F, Lload, R_HFI, C_hi_val)
            for fs in f_scan])
        f_peak_actual = f_scan[int(np.argmax(ratio_scan))]
        if f_peak_actual < f0_est * 0.95:
            return 1e-17
        return best_cs

    def _joint_optimise_cs(self, params, n_active, f0, Rload, Cload_F, Lload,
                           R_HFI=0.0, C_hi_val=0.0):
        """
        Coordinate-descent refinement of Cs[0..n_active-1].
        R_HFI and C_hi_val must be forwarded so each step uses the correct
        effective parallel capacitance (C_load || C_hi when R_HFI ≈ 0).
        """
        MAX_ROUNDS = 2 if self._fast_eval else 4
        for _ in range(MAX_ROUNDS):
            prev = params[:n_active, 5].copy()
            for i in range(n_active):
                harm_i = params[i, 7]
                params[i, 5] = self._cs_for_max_ratio(
                    params[:n_active], i, harm_i * f0,
                    Rload, Cload_F, Lload, R_HFI, C_hi_val)
            if np.all(np.abs(params[:n_active, 5] - prev) /
                      np.maximum(prev, 1e-20) < 0.005):
                break

    def _params_from_resonators(self):
        """
        Build a numpy array (N, 8) from enabled resonator tabs.
        Also stores self._res_index_map: list of resonator tab indices for each row,
        so _write_cs_back can write back to the correct tab regardless of which
        resonators are enabled.
        Columns: R1(Ω), C1(F), L(H), k, R_DC(Ω), Cs(F), Cp(F), harm#
        """
        rows = []
        index_map = []
        if not self.v_global_res_enable.get():
            self._res_index_map = []
            return np.empty((0, 8))
        for i, res in enumerate(self.resonators):
            if not res['enabled'].get():
                continue
            try:
                harm_num = int(res['harm'].get())
            except Exception:
                harm_num = i + 1
            try:
                rows.append([
                    float(res['r1'].get()),
                    float(res['c1'].get()),
                    float(res['l'].get()),
                    float(res['k'].get()),
                    float(res['rdc'].get()),
                    float(res['cs'].get()),
                    float(res['cp'].get()),
                    float(harm_num),
                ])
                index_map.append(i)
            except ValueError:
                pass
        self._res_index_map = index_map
        return np.array(rows) if rows else np.empty((0, 8))

    def _run_cs_optimization(self):
        """
        Tune Cs for each active resonator so its anti-resonance falls at
        f_base * (k+1) for resonator k.  Protocol identical to
        Res_Opt_v1_34._run_cs_optimization / _joint_optimise_cs.
        Runs in a background thread; plots |Iload/Iser| and |Z| live.
        """
        params = self._params_from_resonators()
        N = len(params)
        if N == 0:
            messagebox.showwarning("No Active Resonators",
                                   "No enabled resonators found.\n\n"
                                   "Check that:\n"
                                   "  • The GLOBAL ENABLE switch is ON\n"
                                   "  • At least one resonator tab is checked")
            return

        try:
            f0       = float(self.v_f.get())   * 1e6        # fundamental Hz
            Rload    = float(self.v_rload.get()) + 1e-12
            Cload_F  = float(self.v_cload.get()) * 1e-12    # Farads
            Lload    = float(self.v_ltl.get())  * 1e-9      # L_TL as Lload
            R_HFI    = float(self.v_rhfi.get())             # NEW: HFI Resistance
            C_hi_val = float(self.v_chi.get()) * 1e-12      # NEW: HFI Capacitance in Farads
        except ValueError:
            messagebox.showerror("Bad values", "Check R_load, C_load, L_TL, R_HFI, and C_hi inputs.")
            return

        max_harm  = int(round(max(params[:, 7])))
        f_MHz  = np.linspace(max(f0 * 0.3, 1e6), 1.1 * max_harm * f0, 2000) / 1e6
        f_plot = f_MHz * 1e6
        colors = plt.cm.tab10(np.linspace(0, 1, max(N, 1)))

        # ── Switch to Cs Tuning tab and use its embedded figure ──────────
        cs_tab_idx = list(self.plot_notebook.tabs()).index(str(self.tab_cs))
        self.plot_notebook.select(cs_tab_idx)

        fig    = self.fig_cs
        canvas = self.canvas_cs
        fig.clf()
        ax_ratio, ax_Z = fig.subplots(2, 1)

        ax_ratio.set_xlabel("Frequency (MHz)")
        ax_ratio.set_ylabel("|Iload / Iser|")
        ax_ratio.set_title("Current Ratio — Cs Optimisation Progress")
        ax_ratio.grid(True, alpha=0.4)
        ax_Z.set_xlabel("Frequency (MHz)")
        ax_Z.set_ylabel("|Z_resonator| (Ω)")
        ax_Z.set_title("Individual Resonator Impedance")
        ax_Z.grid(True, alpha=0.4)
        ax_Z.set_yscale('log')

        ghost_lines = []; bold_lines = []; Z_lines = []
        harm_vlines_ratio = []; harm_vlines_Z = []
        for k in range(N):
            harm_k = int(round(params[k, 7]))
            gl, = ax_ratio.plot([], [], color='#cccccc', lw=0.8, zorder=1, visible=False)
            bl, = ax_ratio.plot([], [], color=colors[k], lw=2.0, zorder=3,
                                visible=False, label=f"Step {k+1}: Res{k+1} harm{harm_k}")
            ghost_lines.append(gl); bold_lines.append(bl)
            zl, = ax_Z.plot([], [], color=colors[k], lw=1.0, alpha=0.4,
                            visible=False, label=f"Res{k+1} harm{harm_k}")
            Z_lines.append(zl)
            harm_vlines_ratio.append(
                ax_ratio.axvline(harm_k * f0 / 1e6, color=colors[k],
                                 ls='--', lw=1.0, alpha=0.0))
            harm_vlines_Z.append(
                ax_Z.axvline(harm_k * f0 / 1e6, color=colors[k],
                             ls=':', lw=1.0, alpha=0.0))

        ax_ratio.set_xlim(f_MHz[0], f_MHz[-1])
        ax_Z.set_xlim(f_MHz[0], f_MHz[-1])
        ax_ratio.set_ylim(0, 1.1)
        ax_Z.set_ylim(1, 1e6)
        ax_ratio.legend(fontsize=7, ncol=2)
        ax_Z.legend(fontsize=7, ncol=2)
        canvas.draw()

        status_var = self._cs_status_var
        status_var.set("Starting…")
        q = queue.Queue()

        def worker():
            p = params.copy()
            failed = []   # list of (tab_index, harm_k, reason)

            for k in range(N):
                harm_k = int(round(p[k, 7]))
                f_target_k = harm_k * f0
                q.put(('status',
                       f"Step {k+1}/{N}: scanning Cs{k+1} for harm {harm_k} "
                       f"({f_target_k/1e6:.2f} MHz)…"))
                result_cs = self._cs_for_max_ratio(
                    p[:k+1], k, f_target_k, Rload, Cload_F, Lload,
                    R_HFI, C_hi_val)
                p[k, 5] = result_cs

                if result_cs < 1e-16:
                    L_k = p[k, 2]; Cp_k = p[k, 6]
                    C_par = Cp_k + Cload_F + C_hi_val
                    omega_k = 2.0 * np.pi * f_target_k
                    c_eff_t = 1.0 / (L_k * omega_k**2) if L_k > 0 else 0
                    tab_i = self._res_index_map[k] if k < len(self._res_index_map) else k
                    if c_eff_t > 0 and C_par <= c_eff_t:
                        # C_par = Cp+Cload+Chi is the total shunt capacitance.
                        # System anti-resonance requires C_par > 1/(L*w^2).
                        min_harm = None
                        for h in range(harm_k + 1, 33):
                            c_t_h = 1.0 / (L_k * (2*np.pi*h*f0)**2)
                            if C_par > c_t_h:
                                min_harm = h
                                break
                        f_ar_max = 1/(2*np.pi*np.sqrt(L_k*C_par)) if C_par > 0 else 0
                        if min_harm:
                            reason = (f"C_par = Cp+Cload+Chi = {C_par*1e12:.1f} pF ≤ "
                                      f"1/(L·ω²) = {c_eff_t*1e12:.1f} pF.\n"
                                      f"    Anti-res first possible at harm {min_harm} "
                                      f"({min_harm*f0/1e6:.1f} MHz).\n"
                                      f"    Assign this resonator to harm ≥ {min_harm}, "
                                      f"add more C_load/Cp, or use smaller L.")
                        else:
                            reason = (f"C_par = {C_par*1e12:.1f} pF ≤ 1/(L·ω²) = "
                                      f"{c_eff_t*1e12:.1f} pF at all harmonics ≤ 32.\n"
                                      f"    Max reachable anti-res ≈ {f_ar_max/1e6:.0f} MHz.\n"
                                      f"    Use a smaller L.")
                    else:
                        reason = "Resonator does not improve current ratio at this harmonic."
                    failed.append((tab_i + 1, harm_k, f_target_k / 1e6, reason))

                snap = p[:k+1].copy()
                ratio_vals = np.array([
                    self._current_ratio_at(f, snap, Rload, Cload_F, Lload, R_HFI, C_hi_val)
                    for f in f_plot])
                Z_curves = [
                    np.array([abs(self._single_res_impedance(f, snap[i]))
                              for f in f_plot])
                    for i in range(len(snap))]
                q.put(('step', k, snap, ratio_vals, Z_curves))

                if k >= 1:
                    q.put(('status',
                           f"Step {k+1}/{N}: joint refinement Cs1..Cs{k+1}…"))
                    self._joint_optimise_cs(p, k + 1, f0, Rload, Cload_F, Lload,
                                            R_HFI, C_hi_val)

                    snap2 = p[:k+1].copy()
                    ratio_vals2 = np.array([
                        self._current_ratio_at(f, snap2, Rload, Cload_F, Lload, R_HFI, C_hi_val)
                        for f in f_plot])
                    Z_curves2 = [
                        np.array([abs(self._single_res_impedance(f, snap2[i]))
                                  for f in f_plot])
                        for i in range(len(snap2))]
                    q.put(('refine', k, snap2, ratio_vals2, Z_curves2))

            q.put(('done', p.copy(), failed))

        ratio_history = []

        def _paint(k, snap, ratio_vals, Z_curves, is_refine=False):
            if is_refine:
                bold_lines[k].set_data(f_MHz, ratio_vals)
                bold_lines[k].set_label(
                    f"Step {k+1}: +L{k+1} Cs={snap[k,5]*1e12:.3g}pF ✓")
                ratio_history[k] = ratio_vals
            else:
                ratio_history.append(ratio_vals)
                for prev_k in range(k):
                    ghost_lines[prev_k].set_data(f_MHz, ratio_history[prev_k])
                    ghost_lines[prev_k].set_visible(True)
                    bold_lines[prev_k].set_visible(False)
                bold_lines[k].set_data(f_MHz, ratio_vals)
                bold_lines[k].set_label(
                    f"Step {k+1}: +L{k+1} Cs={snap[k,5]*1e12:.3g}pF")
                bold_lines[k].set_visible(True)
                for i in range(k + 1):
                    harm_vlines_ratio[i].set_alpha(0.55)
                    harm_vlines_Z[i].set_alpha(0.45)

            all_vals = np.concatenate(ratio_history)
            ax_ratio.set_ylim(0, max(all_vals.max() * 1.15, 0.1))

            for i, Z_vals in enumerate(Z_curves):
                lw    = 2.0 if i == k else 1.0
                alpha = 1.0 if i == k else 0.55
                Z_lines[i].set_data(f_MHz, Z_vals)
                Z_lines[i].set_linewidth(lw)
                Z_lines[i].set_alpha(alpha)
                Z_lines[i].set_label(f"L{i+1} Cs={snap[i,5]*1e12:.3g}pF")
                Z_lines[i].set_visible(True)

            z_flat = np.concatenate(Z_curves)
            z_flat = z_flat[np.isfinite(z_flat) & (z_flat > 0)]
            if len(z_flat):
                ax_Z.set_ylim(z_flat.min() * 0.5, z_flat.max() * 3)

            ax_ratio.legend(fontsize=7, ncol=2)
            ax_Z.legend(fontsize=7, ncol=2)
            canvas.draw()
            self.root.update()

        def poll():
            try:
                msg = q.get_nowait()
                if msg[0] == 'status':
                    status_var.set(msg[1])
                    self.root.update_idletasks()
                elif msg[0] in ('step', 'refine'):
                    _, k, snap, ratio_vals, Z_curves = msg
                    self._write_cs_back(snap)
                    _paint(k, snap, ratio_vals, Z_curves,
                           is_refine=(msg[0] == 'refine'))
                elif msg[0] == 'done':
                    _, final, failed = msg
                    self._write_cs_back(final)
                    self._cs_optimized = True
                    self._led_stop()
                    status_var.set(
                        "Cs optimisation complete: " +
                        "  ".join([f"L{i+1}={final[i,5]*1e12:.3g}pF"
                                   for i in range(N)]))
                    self.calculate_and_plot()
                    self.run_ac_sweep()   # update transfer function and current ratio with new Cs

                    if failed:
                        lines = ["Cs optimisation could not find a valid Cs for:\n"]
                        for tab_n, harm_k, f_mhz, reason in failed:
                            lines.append(f"  Resonator {tab_n}  (harm {harm_k}, {f_mhz:.2f} MHz):")
                            lines.append(f"    {reason}\n")
                        messagebox.showwarning("Cs Optimisation — Failed Resonators",
                                               "\n".join(lines))
                    return
            except queue.Empty:
                pass
            self.root.after(20, poll)

        self._led_start()
        threading.Thread(target=worker, daemon=True).start()
        self.root.after(50, poll)

    def _run_rmo_lch_optimizer(self):
        """
        2D grid search over R_mo and L_choke to maximise FOM/N.
        FOM/N = 1/N_act - 1/N_base  where N = E_drawn / ½CV²f.
        Runs in a background thread; results shown in a dedicated window
        and stored in self.last_rmo_lch_opt for inclusion in parameters file.
        """
        import threading, queue as _queue

        # ── snapshot current circuit params (resonators fixed) ───────────
        try:
            f_base     = float(self.v_f.get()) * 1e6
            N_harm     = int(self.v_n.get())
            R_load     = float(self.v_rload.get()) + 1e-12
            C_load_val = float(self.v_cload.get())
            L_TL_val   = float(self.v_ltl.get()) * 1e-9
            R_hfi      = float(self.v_rhfi.get()) + 1e-12
            C_hi_val   = float(self.v_chi.get())
            R_gnd      = self._get_rgnd()
            mo_harms   = self.parse_mo_harms()
        except Exception as e:
            messagebox.showerror("Error", f"Could not read circuit params: {e}")
            return

        active_resonators = []
        if self.v_global_res_enable.get():
            for res in self.resonators:
                if res['enabled'].get():
                    try:
                        active_resonators.append((
                            float(res['r1'].get()), float(res['c1'].get()),
                            float(res['l'].get()),  float(res['k'].get()),
                            float(res['rdc'].get()), float(res['cs'].get()),
                            float(res['cp'].get())))
                    except ValueError:
                        pass

        if not self.custom_loaded:
            delay = float(self.v_delay.get()); ramp = float(self.v_ramp.get())
            up    = float(self.v_up.get());    vmin = float(self.v_vmin.get())
            vmax  = float(self.v_vmax.get())
            mags, phases = self.generate_trapezoid(f_base, N_harm, delay, ramp, up, vmin, vmax)
        else:
            mags   = self.target_mags[:N_harm+1]
            phases = self.target_phases[:N_harm+1]
            if len(mags) < N_harm+1:
                mags   = np.pad(mags,   (0, N_harm+1-len(mags)))
                phases = np.pad(phases, (0, N_harm+1-len(phases)))
            vmin = float(self.v_vmin.get()); vmax = float(self.v_vmax.get())

        V_node_req = mags * np.exp(1j*phases)
        w = 2 * np.pi * np.arange(N_harm+1) * f_base

        # ── ask for R_min constraint ─────────────────────────────────────
        from tkinter.simpledialog import askfloat
        R_min = askfloat(
            "R_gen minimum",
            "Minimum allowed R_gen (Ω):\n"
            "(Constrains the search to realistic driver impedances.\n"
            " Set to 0 for unconstrained.)",
            initialvalue=50.0, minvalue=0.0, maxvalue=10000.0,
            parent=self.root)
        if R_min is None:
            return   # user cancelled

        C_total_F  = (float(self.v_cload.get()) + float(self.v_chi.get())) * 1e-12
        V_swing    = float(vmax) - float(vmin)
        half_cv2_f = 0.5 * C_total_F * V_swing**2 * f_base

        # ── result window ────────────────────────────────────────────────
        win = tk.Toplevel(self.root)
        win.title(f"R_gen / L_choke Optimizer — Results  (R_min={R_min:.1f}Ω)")
        win.geometry("520x380")
        win.resizable(True, True)

        status_var = tk.StringVar(value="Starting search…")
        ttk.Label(win, textvariable=status_var, relief="sunken",
                  anchor="w").pack(side=tk.BOTTOM, fill=tk.X)

        btn_frame = ttk.Frame(win)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=4)
        apply_btn = ttk.Button(btn_frame, text="Apply optimal values to circuit",
                               state='disabled')
        apply_btn.pack(side=tk.LEFT, padx=8)
        ttk.Button(btn_frame, text="Close", command=win.destroy).pack(side=tk.RIGHT, padx=8)

        txt = tk.Text(win, font=('Courier', 10), state='disabled', wrap='none')
        sb  = ttk.Scrollbar(win, command=txt.yview)
        txt.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        txt.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        def append(s):
            txt.config(state='normal')
            txt.insert(tk.END, s + '\n')
            txt.see(tk.END)
            txt.config(state='disabled')

        q = _queue.Queue()

        def worker():
            def eval_fom_n(R_mo, L_ch):
                try:
                    L_use = max(L_ch, 1e-21)
                    rb = self._sim_core([], f_base, N_harm, R_load, C_load_val,
                                        L_TL_val, R_mo, L_use, R_hfi, C_hi_val,
                                        mo_harms, V_node_req, w, R_gnd=R_gnd)
                    ra = self._sim_core(active_resonators, f_base, N_harm, R_load,
                                        C_load_val, L_TL_val, R_mo, L_use, R_hfi,
                                        C_hi_val, mo_harms, V_node_req, w, R_gnd=R_gnd)
                    p_a = ra['p_in_avg']; p_b = rb['p_in_avg']
                    if p_a <= 0 or p_b <= 0 or half_cv2_f <= 0:
                        return -np.inf
                    return 1.0/(p_a/half_cv2_f) - 1.0/(p_b/half_cv2_f)
                except Exception:
                    return -np.inf

            # Coarse grid — R_mo constrained to >= R_min
            R_lo_coarse = max(R_min, 0.5)
            R_hi_coarse = max(R_lo_coarse * 10, 200.0)
            R_vals = np.logspace(np.log10(R_lo_coarse), np.log10(R_hi_coarse), 18)
            L_vals = np.concatenate([[0.0], np.logspace(np.log10(1e-9), np.log10(500e-9), 17)])
            total  = len(R_vals) * len(L_vals)
            done   = 0; best_val = -np.inf; best_R = None; best_L = None

            for R in R_vals:
                for L in L_vals:
                    v = eval_fom_n(R, L)
                    done += 1
                    if v > best_val:
                        best_val = v; best_R = R; best_L = L
                    q.put(('progress', done, total, best_R, best_L, best_val))

            # Fine grid around best coarse point
            q.put(('status', "Refining around best point…"))
            R_fine = np.logspace(np.log10(max(best_R*0.5, R_min, 0.1)),
                                 np.log10(min(best_R*2.0, 500)), 20)
            L_lo   = max(best_L*0.3, 0.0)
            L_hi   = best_L*3.0 if best_L > 0 else 20e-9
            L_fine = np.concatenate([[0.0], np.logspace(np.log10(max(L_lo, 1e-10)),
                                                         np.log10(L_hi), 19)])
            total2 = len(R_fine)*len(L_fine); done2 = 0

            for R in R_fine:
                for L in L_fine:
                    v = eval_fom_n(R, L)
                    done2 += 1
                    if v > best_val:
                        best_val = v; best_R = R; best_L = L
                    q.put(('progress2', done2, total2, best_R, best_L, best_val))

            q.put(('done', best_R, best_L, best_val))

        def poll():
            try:
                while True:
                    msg = q.get_nowait()
                    if msg[0] == 'progress':
                        _, done, total, bR, bL, bv = msg
                        status_var.set(
                            f"Coarse {done}/{total}  best FOM/N={bv:.5f}"
                            f"  R={bR:.2f}Ω  L={bL*1e9:.1f}nH")
                    elif msg[0] == 'progress2':
                        _, done, total, bR, bL, bv = msg
                        status_var.set(
                            f"Fine {done}/{total}  best FOM/N={bv:.5f}"
                            f"  R={bR:.2f}Ω  L={bL*1e9:.1f}nH")
                    elif msg[0] == 'status':
                        status_var.set(msg[1])
                    elif msg[0] == 'done':
                        _, best_R, best_L, best_val = msg

                        # Full metrics at optimum
                        L_use = max(best_L, 1e-21)
                        rb = self._sim_core([], f_base, N_harm, R_load, C_load_val,
                                            L_TL_val, best_R, L_use, R_hfi, C_hi_val,
                                            mo_harms, V_node_req, w, R_gnd=R_gnd)
                        ra = self._sim_core(active_resonators, f_base, N_harm, R_load,
                                            C_load_val, L_TL_val, best_R, L_use, R_hfi,
                                            C_hi_val, mo_harms, V_node_req, w, R_gnd=R_gnd)
                        p_a = ra['p_in_avg']; p_b = rb['p_in_avg']
                        N_a = p_a/half_cv2_f; N_b = p_b/half_cv2_f
                        fom_opt  = 1 - p_a/p_b if p_b > 0 else 0
                        fomn_opt = 1/N_a - 1/N_b

                        # Current circuit metrics for comparison
                        R_cur = float(self.v_rmo.get()) + 1e-12
                        L_cur = float(self.v_lch.get()) * 1e-9 + 1e-21
                        rb_c = self._sim_core([], f_base, N_harm, R_load, C_load_val,
                                              L_TL_val, R_cur, L_cur, R_hfi, C_hi_val,
                                              mo_harms, V_node_req, w, R_gnd=R_gnd)
                        ra_c = self._sim_core(active_resonators, f_base, N_harm, R_load,
                                              C_load_val, L_TL_val, R_cur, L_cur, R_hfi,
                                              C_hi_val, mo_harms, V_node_req, w, R_gnd=R_gnd)
                        p_ac = ra_c['p_in_avg']; p_bc = rb_c['p_in_avg']
                        fom_cur  = 1 - p_ac/p_bc if p_bc > 0 else 0
                        fomn_cur = (1/(p_ac/half_cv2_f) - 1/(p_bc/half_cv2_f)
                                    if p_ac > 0 and p_bc > 0 else 0)
                        improv = fomn_opt/fomn_cur if fomn_cur > 0 else float('inf')

                        # Store for parameters file
                        self.last_rmo_lch_opt = {
                            'R_min_constraint': R_min,
                            'best_R_mo':    best_R,
                            'best_L_ch_nH': best_L * 1e9,
                            'fom_opt':      fom_opt,
                            'N_opt':        N_a,
                            'fomn_opt':     fomn_opt,
                            'e_drawn_opt_pJ': p_a / f_base * 1e12,
                            'e_base_opt_pJ':  p_b / f_base * 1e12,
                            'fom_cur':      fom_cur,
                            'N_cur':        p_ac/half_cv2_f,
                            'fomn_cur':     fomn_cur,
                            'e_drawn_cur_pJ': p_ac / f_base * 1e12,
                            'improvement':  improv,
                            'R_cur':        float(self.v_rmo.get()),
                            'L_cur_nH':     float(self.v_lch.get()),
                        }

                        # Print results
                        append(f"R_gen / L_choke Optimizer Results")
                        append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        append(f"Constraint: R_gen >= {R_min:.1f} Ω")
                        append("=" * 50)
                        append("")
                        append("  OPTIMAL VALUES")
                        append(f"  {'R_gen':20s} = {best_R:.4f} Ω")
                        append(f"  {'L_choke':20s} = {best_L*1e9:.4f} nH")
                        append("")
                        append("  METRICS AT OPTIMUM")
                        append(f"  {'Recycl. FOM':20s} = {fom_opt:.4f}")
                        append(f"  {'Overhead N':20s} = {N_a:.3f} × ½CV²")
                        append(f"  {'FOM/N':20s} = {fomn_opt:.5f}  ← maximised")
                        append(f"  {'E_drawn':20s} = {p_a/f_base*1e12:.3f} pJ/cycle")
                        append(f"  {'E_drawn (baseline)':20s} = {p_b/f_base*1e12:.3f} pJ/cycle")
                        append("")
                        append("  CURRENT CIRCUIT (for comparison)")
                        append(f"  {'R_gen':20s} = {float(self.v_rmo.get()):.4f} Ω")
                        append(f"  {'L_choke':20s} = {float(self.v_lch.get()):.4f} nH")
                        append(f"  {'Recycl. FOM':20s} = {fom_cur:.4f}")
                        append(f"  {'Overhead N':20s} = {p_ac/half_cv2_f:.3f} × ½CV²")
                        append(f"  {'FOM/N':20s} = {fomn_cur:.5f}")
                        append(f"  {'E_drawn':20s} = {p_ac/f_base*1e12:.3f} pJ/cycle")
                        append("")
                        append(f"  FOM/N improvement: {improv:.2f}×")
                        append("=" * 50)
                        status_var.set("Done. Click 'Apply' to use optimal values.")

                        def apply_opt(R=best_R, L=best_L):
                            self.v_rmo.set(f"{R:.4f}")
                            self.v_lch.set(f"{L*1e9:.4f}")
                            self.calculate_and_plot()
                            win.destroy()

                        apply_btn.config(state='normal', command=apply_opt)
                        return
            except _queue.Empty:
                pass
            win.after(80, poll)

        threading.Thread(target=worker, daemon=True).start()
        win.after(80, poll)

    def _check_ar_overshoot(self):
        """
        After Cs optimisation: check if any resonator's anti-resonance peak
        is above its target frequency.  Uses the FULL combined Y_res_total
        (all resonators together) exactly as run_ac_sweep does, so the ratio
        matches what is shown in the frequency sweep plot.
        Only fires if a genuine local peak (ratio > 1) in a ±20% window around
        f_target sits more than 1% above f_target.
        """
        try:
            f0     = float(self.v_f.get()) * 1e6
            C_load = float(self.v_cload.get()) * 1e-12
            C_hi   = float(self.v_chi.get())   * 1e-12
            Rload  = float(self.v_rload.get())  + 1e-12
            Lload  = float(self.v_ltl.get())    * 1e-9
            R_HFI  = float(self.v_rhfi.get())   + 1e-12
        except Exception:
            return
        if not self.v_global_res_enable.get():
            return

        # Collect all enabled resonators
        active = []
        active_harms = []
        res_indices = []
        for i, res in enumerate(self.resonators):
            if not res['enabled'].get():
                continue
            try:
                active.append((
                    float(res['l'].get()),   float(res['cs'].get()),
                    float(res['cp'].get()),  float(res['r1'].get()),
                    float(res['c1'].get()),  float(res['k'].get()),
                    float(res['rdc'].get()), int(res['harm'].get())))
                res_indices.append(i)
            except Exception:
                pass

        if not active:
            return

        problems = []
        for idx_a, (L, Cs, Cp, R1, C1, k_v, R_DC, harm) in enumerate(active):
            if L <= 0 or Cs <= 0:
                continue
            f_target = harm * f0
            f_lo = f_target * 0.80
            f_hi = f_target * 1.20
            f_scan = np.linspace(f_lo, f_hi, 400)
            om     = 2 * np.pi * f_scan

            # Combined Y_res_total from ALL resonators (matches run_ac_sweep)
            Y_res_total = np.zeros(len(f_scan), dtype=complex)
            for (L2, Cs2, Cp2, R12, C12, k2, R_DC2, _) in active:
                R_var2 = k2 * np.sqrt(f_scan)
                Z_A2   = R12 + 1.0/(1j*om*C12) if C12 > 0 else np.full(len(f_scan), R12+0j)
                Z_B2   = (R_var2 + R_DC2) + 1j*om*L2
                Z_AB2  = 1.0/(1.0/Z_A2 + 1.0/Z_B2)
                if Cs2 > 0:
                    Y_s2 = 1.0/(Z_AB2 + 1.0/(1j*om*Cs2))
                else:
                    Y_s2 = np.zeros(len(f_scan), dtype=complex)
                Y_res_total += Y_s2 + 1j*om*Cp2

            Z_ld  = Rload + 1j*om*Lload + 1.0/(1j*om*C_load)
            Y_ld  = 1.0/Z_ld
            Y_hfi = (1.0/(R_HFI + 1.0/(1j*om*C_hi))
                     if C_hi > 1e-15 else np.zeros(len(f_scan), dtype=complex))
            ratio = np.abs(Y_ld) / (np.abs(Y_ld + Y_hfi + Y_res_total) + 1e-30)

            idx_peak = int(np.argmax(ratio))
            peak_val = ratio[idx_peak]
            f_peak   = f_scan[idx_peak]

            if peak_val > 1.0 and f_peak > f_target * 1.01:
                problems.append(
                    f"  Resonator {res_indices[idx_a]+1}  "
                    f"(harm {harm}, target {f_target/1e6:.3f} MHz):\n"
                    f"    Peak |I_load/I_choke| = {peak_val:.2f} "
                    f"at {f_peak/1e6:.3f} MHz  "
                    f"({(f_peak/f_target-1)*100:.1f}% above target)\n"
                    f"    Cs = {Cs*1e12:.2f} pF  →  increase Cs to shift peak down.")

        if problems:
            messagebox.showwarning(
                "Peak Ratio Above Target Frequency",
                "The following resonators peak above their target harmonic.\n"
                "Energy recycling will occur at the wrong frequency:\n\n" +
                "\n".join(problems))

    def _write_cs_back(self, params):
        """
        Write optimised Cs values back into the correct resonator Entry widgets.
        Skips resonators where Cs = 1e-17 (optimization failed) to preserve
        the existing Cs value rather than overwriting with a near-zero nonsense value.
        """
        index_map = getattr(self, '_res_index_map', list(range(len(params))))
        for row_i, tab_i in enumerate(index_map):
            if row_i >= len(params):
                break
            cs_val = params[row_i, 5]
            if cs_val < 1e-16:
                continue   # optimization failed — leave previous Cs untouched
            if tab_i < len(self.resonators):
                try:
                    self.resonators[tab_i]['cs'].set(f"{cs_val:.3e}")
                except Exception:
                    pass

    # ── LED helpers ─────────────────────────────────────────────────────

    def _led_start(self):
        self._led_on = True
        self._led_flash()

    def _led_stop(self):
        self._led_on = False
        if self._led_job:
            self.root.after_cancel(self._led_job)
            self._led_job = None
        try:
            self._led_canvas.itemconfig(self._led, fill="#cccccc")
        except Exception:
            pass

    def _led_flash(self):
        if not self._led_on:
            return
        try:
            cur = self._led_canvas.itemcget(self._led, 'fill')
            self._led_canvas.itemconfig(
                self._led, fill="#00cc44" if cur != "#00cc44" else "#005522")
        except Exception:
            pass
        self._led_job = self.root.after(400, self._led_flash)

    # ═══════════════════════════════════════════════════════════════════════
    # RESONATOR TABS  (unchanged from V34)
    # ═══════════════════════════════════════════════════════════════════════

    def load_resonators(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("Resonator Files", "*.res"), ("All Files", "*.*")])
        if not filepath:
            return
        try:
            res_data = []
            harm_nums = []
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split(',')
                    if len(parts) >= 7:
                        res_data.append((float(parts[0]), float(parts[1]),
                                         float(parts[2]), float(parts[3]),
                                         float(parts[4]), float(parts[5]),
                                         float(parts[6])))
                        # column 8 = harm# (added in V38); default to i+1 if absent
                        harm_nums.append(int(float(parts[7])) if len(parts) >= 8
                                         else len(res_data))
            if res_data:
                self.build_resonator_tabs(res_data)
                # Apply saved harm numbers
                for res, h in zip(self.resonators, harm_nums):
                    res['harm'].set(h)
                self.update_fres()
                self.calculate_and_plot()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load resonators: {str(e)}")

    def save_resonators(self):
        filepath = filedialog.asksaveasfilename(
            defaultextension=".res",
            filetypes=[("Resonator Files", "*.res"), ("All Files", "*.*")])
        if not filepath:
            return
        try:
            lines = ["# R1(Ohm), C1(F), L(H), k, R_DC(Ohm), Cs(F), Cp(F), harm#\n"]
            for res in self.resonators:
                if not res['enabled'].get():
                    continue
                try:
                    harm_n = int(res['harm'].get())
                except Exception:
                    harm_n = 1
                row = (f"{float(res['r1'].get()):.6e},"
                       f"{float(res['c1'].get()):.6e},"
                       f"{float(res['l'].get()):.6e},"
                       f"{float(res['k'].get()):.6e},"
                       f"{float(res['rdc'].get()):.6e},"
                       f"{float(res['cs'].get()):.6e},"
                       f"{float(res['cp'].get()):.6e},"
                       f"{harm_n}\n")
                lines.append(row)
            with open(filepath, 'w') as f:
                f.writelines(lines)
            messagebox.showinfo("Saved",
                                f"Saved {len(lines)-1} resonator(s) to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save resonators: {str(e)}")

    def build_resonator_tabs(self, res_data_list):
        for tab in self.notebook_res.tabs():
            self.notebook_res.forget(tab)
        self.resonators = []

        def fmt(val):
            return f"{val:.3e}"

        for i, data in enumerate(res_data_list):
            frame = ttk.Frame(self.notebook_res, padding="5")
            self.notebook_res.add(frame, text=f"Res {i+1}")
            var_enabled = tk.BooleanVar(value=True)

            # Top row: enable checkbox + harmonic assignment
            top_row = ttk.Frame(frame)
            top_row.grid(row=0, column=0, columnspan=4, sticky=tk.W, pady=2)
            ttk.Checkbutton(top_row, text=f"Include Resonator {i+1}",
                            variable=var_enabled).pack(side=tk.LEFT)
            ttk.Label(top_row, text="   Target harmonic:").pack(side=tk.LEFT)
            var_harm = tk.IntVar(value=i+1)
            harm_spin = ttk.Spinbox(top_row, from_=1, to=32, width=4,
                                    textvariable=var_harm)
            harm_spin.pack(side=tk.LEFT, padx=(3, 0))
            var_harm.trace_add("write", self.update_fres)

            def add_entry_res(parent, label_text, default_val, row, col,
                              trace_func=None):
                ttk.Label(parent, text=label_text).grid(
                    row=row, column=col, sticky=tk.W, pady=2, padx=(0, 5))
                var = tk.StringVar(value=str(default_val))
                if trace_func:
                    var.trace_add("write", trace_func)
                entry = ttk.Entry(parent, textvariable=var, width=11)
                entry.grid(row=row, column=col + 1, sticky=tk.EW,
                           pady=2, padx=(0, 10))
                return var

            var_r1  = add_entry_res(frame, "R1 (Ω):",    fmt(data[0]), 1, 0, self.update_fres)
            var_c1  = add_entry_res(frame, "C1 (F):",    fmt(data[1]), 1, 2, self.update_fres)
            var_l   = add_entry_res(frame, "L (H):",     fmt(data[2]), 2, 0, self.update_fres)
            var_k   = add_entry_res(frame, "k (Skin):",  fmt(data[3]), 2, 2, self.update_fres)
            var_rdc = add_entry_res(frame, "R_DC (Ω):",  fmt(data[4]), 3, 0, self.update_fres)
            var_cs  = add_entry_res(frame, "C_s (F):",   fmt(data[5]), 3, 2, self.update_fres)
            var_cp  = add_entry_res(frame, "C_p (F):",   fmt(data[6]), 4, 0, self.update_fres)

            lbl_fres = ttk.Label(frame, text="Calculating…", foreground="blue")
            lbl_fres.grid(row=5, column=0, columnspan=4, pady=(5, 4))

            self.resonators.append({
                'enabled': var_enabled,
                'harm':    var_harm,
                'r1': var_r1, 'c1': var_c1, 'l': var_l, 'k': var_k,
                'rdc': var_rdc, 'cs': var_cs, 'cp': var_cp,
                'lbl': lbl_fres,
            })

    def update_fres(self, *args):
        """
        Display Anti-Resonance for each resonator tab.
        Series F_res line is replaced by the peak info after simulation runs.
        Anti-resonance formula: f = 1/(2π√(L · Cs·(Cp+Cload+Chi)/(Cs+Cp+Cload+Chi)))
        """
        try:
            cload_F = float(self.v_cload.get()) * 1e-12
        except Exception:
            cload_F = 50e-12
        try:
            chi_F = float(self.v_chi.get()) * 1e-12
        except Exception:
            chi_F = 0.0

        for res in self.resonators:
            try:
                harm_n = int(res['harm'].get())
            except Exception:
                harm_n = 1
            try:
                l  = float(res['l'].get())
                cs = float(res['cs'].get())
                cp = float(res['cp'].get())
                if l > 0 and cs > 0:
                    f_ser = 1.0 / (2.0 * np.pi * np.sqrt(l * cs))
                    # System anti-resonance: Cp || Cload || Chi all in parallel with
                    # the series motional branch. Occurs where C_par > 1/(L*w^2).
                    c_par = cp + cload_F + chi_F
                    omega_harm = 2.0 * np.pi * float(self.v_f.get()) * 1e6 * harm_n
                    c_eff_t = 1.0 / (l * omega_harm**2) if omega_harm > 0 else 0
                    if c_par > c_eff_t > 0:
                        res['lbl'].config(
                            text=f"Harm {harm_n} | Series F_res: {f_ser/1e6:.3f} MHz  |  Peak: --",
                            foreground="blue")
                    elif c_eff_t > 0:
                        f_ar_max = 1.0/(2.0*np.pi*np.sqrt(l*c_par)) if c_par > 0 else 0
                        res['lbl'].config(
                            text=f"Harm {harm_n} | Series F_res: {f_ser/1e6:.3f} MHz  |  "
                                 f"⚠ Anti-res impossible at harm {harm_n} "
                                 f"(C_par={c_par*1e12:.1f}pF < {c_eff_t*1e12:.1f}pF)  "
                                 f"max reachable: {f_ar_max/1e6:.0f}MHz",
                            foreground="red")
                    else:
                        res['lbl'].config(
                            text=f"Harm {harm_n} | Series F_res: {f_ser/1e6:.3f} MHz",
                            foreground="blue")
                else:
                    res['lbl'].config(text=f"Harm {harm_n} | F_res: N/A", foreground="blue")
            except Exception:
                pass

    # ═══════════════════════════════════════════════════════════════════════
    # SIMULATION CORE  (unchanged from V34)
    # ═══════════════════════════════════════════════════════════════════════

    def parse_mo_harms(self):
        raw   = self.v_mo_harms.get().strip()
        harms = set()
        for part in raw.split(','):
            part = part.strip()
            if '-' in part:
                start, end = map(int, part.split('-'))
                harms.update(range(start, end + 1))
            else:
                harms.add(int(part))
        return harms

    def _get_rgnd(self):
        """Read R_toGND, validate >= 1 Ω, return value in Ohms. Returns 1e10 on error."""
        if not hasattr(self, 'v_rgnd'):
            return 1e10
        try:
            v = float(self.v_rgnd.get())
        except (ValueError, tk.TclError):
            return 1e10
        if v < 1.0:
            messagebox.showwarning(
                "R_toGND too small",
                f"R_toGND = {v} Ω would short the MO output to ground.\n"
                "Value must be ≥ 1 Ω. Reverting to 1e10 Ω.")
            self.v_rgnd.set("1e10")
            return 1e10
        return v

    def _update_hfi_harms_label(self, *args):
        """Compute which harmonics HFI covers (all not in mo_harms, 1..N) and display."""
        if not hasattr(self, 'lbl_hfi_harms'):
            return
        try:
            N = int(self.v_n.get())
            mo = self.parse_mo_harms()
        except Exception:
            self.lbl_hfi_harms.config(text="(parse error)", foreground="red")
            return
        hfi = sorted(n for n in range(1, N+1) if n not in mo)
        # DC always stays with MO; warn if 0 is missing from mo_harms
        warn = "  ⚠ DC(0) not in MO — will be missing!" if 0 not in mo else ""
        if hfi:
            # Compress consecutive runs for readability: 2,4,6,8 → 2,4,6,8 (sparse, show all)
            # but 2,3,4,5,6 → 2-6
            def compress(lst):
                if not lst: return ""
                out, s, e = [], lst[0], lst[0]
                for v in lst[1:]:
                    if v == e + 1:
                        e = v
                    else:
                        out.append(str(s) if s == e else f"{s}-{e}")
                        s = e = v
                out.append(str(s) if s == e else f"{s}-{e}")
                return ", ".join(out)
            try:
                c_hi = float(self.v_chi.get()) if hasattr(self, 'v_chi') else 0
            except (ValueError, tk.TclError):
                c_hi = 0
            active = c_hi > 0
            status = compress(hfi)
            color  = "darkblue" if active else "gray"
            suffix = "" if active else "  (C_hi=0, HFI disabled)"
            self.lbl_hfi_harms.config(
                text=f"harmonics {status}{suffix}{warn}", foreground=color)
        else:
            self.lbl_hfi_harms.config(
                text=f"none — MO covers all{warn}", foreground="darkgreen")

    def generate_trapezoid(self, f, N_harm, delay_pct, ramp_pct, up_pct,
                           vmin, vmax):
        T        = 1 / f
        N_samples = 8192
        t        = np.linspace(0, T, N_samples, endpoint=False)
        t_delay  = delay_pct * T
        t_ramp   = ramp_pct  * T
        t_up     = up_pct    * T

        v_targ = np.piecewise(t, [
            t < t_delay,
            (t >= t_delay) & (t < t_delay + t_ramp),
            (t >= t_delay + t_ramp) & (t < t_delay + t_ramp + t_up),
            (t >= t_delay + t_ramp + t_up) & (t < t_delay + 2*t_ramp + t_up)],
            [vmin,
             lambda x: vmin + (vmax-vmin)*((x-t_delay)/t_ramp),
             vmax,
             lambda x: vmax - (vmax-vmin)*((x-(t_delay+t_ramp+t_up))/t_ramp),
             vmin])
        c_targ = rfft(v_targ) / N_samples * 2
        c_targ[0] /= 2
        return np.abs(c_targ[:N_harm+1]), np.angle(c_targ[:N_harm+1])

    def get_Ls_impedance_single(self, f, R_DC, R1, C1, L, k):
        if f == 0:
            return R_DC
        omega     = 2 * np.pi * f
        R_var     = k * np.sqrt(f)
        Z_branch1 = R_var + 1j * omega * L
        if C1 > 0:
            Z_branch2 = R1 + 1 / (1j * omega * C1)
            Z_p = (Z_branch1 * Z_branch2) / (Z_branch1 + Z_branch2)
        else:
            Z_p = Z_branch1
        return R_DC + Z_p

    def get_Ls_impedance_array(self, f_arr, R_DC, R1, C1, L, k):
        omega     = 2 * np.pi * f_arr
        R_var     = k * np.sqrt(f_arr)
        Z_branch1 = R_var + 1j * omega * L
        Z_p       = np.zeros_like(f_arr, dtype=complex)
        ac_mask   = (f_arr != 0)
        if C1 > 0:
            Z_branch2       = R1 + 1 / (1j * omega[ac_mask] * C1)
            Z_p[ac_mask]    = (Z_branch1[ac_mask] * Z_branch2) / \
                               (Z_branch1[ac_mask] + Z_branch2)
        else:
            Z_p[ac_mask] = Z_branch1[ac_mask]
        Z_tot           = np.full_like(f_arr, R_DC, dtype=complex)
        Z_tot[ac_mask]  = R_DC + Z_p[ac_mask]
        return Z_tot

    def _sim_core(self, active_resonators, f_base, N_harm,
                  R_load, C_load_val, L_TL_val, R_mo, L_ch, R_hfi, C_hi_val,
                  mo_harms, V_node_req, w, predistort=True, R_gnd=1e10):

        V_mo_gen  = np.zeros(N_harm+1, dtype=complex)
        V_hfi_gen = np.zeros(N_harm+1, dtype=complex)
        Y_mo_arr  = np.zeros(N_harm+1, dtype=complex)
        Y_hfi_arr = np.zeros(N_harm+1, dtype=complex)
        Y_tot_arr = np.zeros(N_harm+1, dtype=complex)
        Y_load_arr = np.zeros(N_harm+1, dtype=complex)
        Y_s_arr = [np.zeros(N_harm+1, dtype=complex) for _ in active_resonators]

        for n in range(N_harm+1):
            f, omega, V_n = self.freqs[n], w[n], V_node_req[n]

            if n == 0:
                Y_load_base = 1/R_load if C_load_val <= 0 else 0j
            else:
                Z_load = R_load + 1j*omega*L_TL_val
                if C_load_val > 0:
                    Z_load += 1 / (1j*omega*(C_load_val*1e-12))
                Y_load_base = 1 / Z_load
            Y_load_arr[n] = Y_load_base

            Y_res_total = 0j
            for idx, (R1, C1, L, k_val, R_DC, C_s, C_p) in enumerate(active_resonators):
                if n != 0:
                    Z_Ls = self.get_Ls_impedance_single(f, R_DC, R1, C1, L, k_val)
                    Y_s  = 1/(Z_Ls + 1/(1j*omega*C_s)) if C_s > 0 else 0j
                    Y_p  = 1j*omega*C_p
                    Y_res_total += (Y_s + Y_p)
                else:
                    Y_s = 0j
                Y_s_arr[idx][n] = Y_s

            Y_load_eff = Y_load_base + Y_res_total
            # C_hi_val > 0: full R_hfi+C_hi branch always (user sets R_hfi, can be 0)
            #   harmonic in mo_harms  → V_hfi_gen=0, branch is passive shunt R_hfi+C_hi
            #   harmonic not in mo_harms → HFI active source
            if C_hi_val <= 0 or n == 0:
                Y_hfi = 0j
            else:
                Y_hfi = 1.0 / (R_hfi + 1.0/(1j*omega*(C_hi_val*1e-12)))
            # MO branch Thevenin equivalent.
            # R_toGND sits at junction X between R_MO and L_choke (diagram-correct).
            # Thevenin from node: Z_th = jω·L_ch + R_mo||R_toGND
            #                     V_th = V_mo_gen × R_toGND/(R_mo + R_toGND)
            R_gnd_safe = max(R_gnd, 1.0)
            R_par = R_mo * R_gnd_safe / (R_mo + R_gnd_safe)   # R_mo || R_toGND
            V_mo_scale = R_gnd_safe / (R_mo + R_gnd_safe)     # voltage divider factor
            Y_mo  = (1/R_par if n == 0
                     else 1/(R_par + 1j*omega*L_ch))
            Y_gnd = 0.0    # already absorbed into R_par above
            Y_tot = Y_load_eff + Y_hfi + Y_mo

            Y_mo_arr[n] = Y_mo; Y_hfi_arr[n] = Y_hfi; Y_tot_arr[n] = Y_tot

            if predistort:
                # Normal mode: solve V_mo_gen so that V_node = V_node_req[n].
                # V_th = V_mo_gen_actual * V_mo_scale, so
                # V_mo_gen_actual = V_th / V_mo_scale = (V_n * Y_tot/Y_mo) / V_mo_scale
                if n in mo_harms:
                    V_mo_gen[n]  = V_n * Y_tot / Y_mo / V_mo_scale
                    V_hfi_gen[n] = 0j
                else:
                    V_mo_gen[n]  = 0j
                    V_hfi_gen[n] = (V_n * Y_tot / Y_hfi
                                    if Y_hfi != 0j else 0j)
            else:
                # No-predistortion: generator outputs V_node_req directly (unscaled).
                if n in mo_harms:
                    V_mo_gen[n]  = V_n / V_mo_scale   # actual output, node sees V_n*V_mo_scale
                    V_hfi_gen[n] = 0j
                else:
                    V_mo_gen[n]  = 0j
                    V_hfi_gen[n] = V_n if Y_hfi != 0j else 0j

        t_plot = np.linspace(0, 2/f_base, 8192, endpoint=False)
        dt     = t_plot[1] - t_plot[0]

        v_targ_t     = np.zeros_like(t_plot)
        v_rec_node_t = np.zeros_like(t_plot)
        v_mo_t  = np.zeros_like(t_plot)
        i_mo_t  = np.zeros_like(t_plot)
        v_hfi_t = np.zeros_like(t_plot)
        i_hfi_t = np.zeros_like(t_plot)
        i_load_t = np.zeros_like(t_plot)
        i_res_s_t = [np.zeros_like(t_plot) for _ in active_resonators]
        V_act_arr = np.zeros(N_harm+1, dtype=complex)

        for n in range(N_harm+1):
            p = np.exp(1j*w[n]*t_plot)
            v_targ_t     += np.real(V_node_req[n] * p)
            v_mo_t       += np.real(V_mo_gen[n]   * p)
            v_hfi_t      += np.real(V_hfi_gen[n]  * p)

            V_act = ((V_mo_gen[n]*Y_mo_arr[n] + V_hfi_gen[n]*Y_hfi_arr[n]) / Y_tot_arr[n]
                     if Y_tot_arr[n] != 0j else 0j)
            V_act_arr[n]  = V_act
            v_rec_node_t += np.real(V_act * p)

            i_mo_t  += np.real((V_mo_gen[n]  - V_act) * Y_mo_arr[n]  * p)
            i_hfi_t += np.real((V_hfi_gen[n] - V_act) * Y_hfi_arr[n] * p)
            i_load_t += np.real(V_act * Y_load_arr[n] * p)
            for idx in range(len(active_resonators)):
                i_res_s_t[idx] += np.real(V_act * Y_s_arr[idx][n] * p)

        p_mo  = v_mo_t  * i_mo_t
        p_hfi = v_hfi_t * i_hfi_t

        p_fwd_mo  = np.maximum(0, p_mo)
        p_fwd_hfi = np.maximum(0, p_hfi)
        p_rev_mo  = np.minimum(0, p_mo)
        p_rev_hfi = np.minimum(0, p_hfi)

        e_dc_drawn_mo_t  = np.cumsum(p_fwd_mo)  * dt
        e_dc_drawn_hfi_t = np.cumsum(p_fwd_hfi) * dt
        e_dc_drawn_t     = e_dc_drawn_mo_t + e_dc_drawn_hfi_t

        e_dumped_heat_t  = np.cumsum(np.abs(p_rev_mo) + np.abs(p_rev_hfi)) * dt
        e_gen_heat_t     = np.cumsum((i_mo_t**2)*R_mo + (i_hfi_t**2)*R_hfi) * dt
        e_load_heat_t    = np.cumsum((i_load_t**2)*R_load) * dt

        p_res_heat_t = np.zeros_like(t_plot)
        for idx, res in enumerate(active_resonators):
            R1, C1, L, k_val, R_DC, C_s, C_p = res
            v_Req_t = np.zeros_like(t_plot)
            for n in range(1, N_harm+1):
                R_eff   = np.real(
                    self.get_Ls_impedance_single(self.freqs[n], R_DC, R1, C1, L, k_val))
                I_n     = V_act_arr[n] * Y_s_arr[idx][n]
                v_Req_t += np.real(I_n * R_eff * np.exp(1j*w[n]*t_plot))
            p_res_heat_t += v_Req_t * i_res_s_t[idx]

        e_res_heat_t    = np.cumsum(p_res_heat_t) * dt
        e_deliv_tank_t  = np.cumsum(v_rec_node_t * (i_mo_t + i_hfi_t)) * dt

        last_time_data = {
            'Time (s)': t_plot, 'V_Target (V)': v_targ_t,
            'V_Node_Recreated (V)': v_rec_node_t,
            'V_MO_Gen (V)': v_mo_t, 'V_HFI_Gen (V)': v_hfi_t,
            'I_MO_Branch (A)': i_mo_t, 'I_HFI_Branch (A)': i_hfi_t,
            'E_DC_Drawn_Total (J)': e_dc_drawn_t,
            'E_DC_Drawn_MO (J)': e_dc_drawn_mo_t,
            'E_DC_Drawn_HFI (J)': e_dc_drawn_hfi_t,
            'E_Dumped_Heat (J)': e_dumped_heat_t,
            'E_Deliv_Tank (J)': e_deliv_tank_t,
            'E_Load_Heat (J)': e_load_heat_t,
            'E_Gen_Heat (J)': e_gen_heat_t,
            'E_Res_Heat (J)': e_res_heat_t
        }

        p_in_mo_avg  = np.mean(p_fwd_mo)
        p_in_hfi_avg = np.mean(p_fwd_hfi)
        p_in_avg     = p_in_mo_avg + p_in_hfi_avg

        p_load_avg       = np.mean(i_load_t**2 * R_load)
        p_dumped_heat_avg = np.mean(np.abs(p_rev_mo) + np.abs(p_rev_hfi))
        p_gen_heat_avg   = np.mean((i_mo_t**2)*R_mo + (i_hfi_t**2)*R_hfi)
        p_res_heat_avg   = np.mean(p_res_heat_t)
        p_mo_heat_avg    = np.mean((i_mo_t**2)*R_mo)
        p_hfi_heat_avg   = np.mean((i_hfi_t**2)*R_hfi)
        p_waste_avg      = p_dumped_heat_avg   # gen I2R already shown separately

        return {
            'V_mo_gen': V_mo_gen, 'V_hfi_gen': V_hfi_gen,
            't_plot': t_plot, 'v_targ_t': v_targ_t,
            'v_rec_node_t': v_rec_node_t,
            'v_mo_t': v_mo_t, 'v_hfi_t': v_hfi_t,
            'i_mo_t': i_mo_t, 'i_hfi_t': i_hfi_t,
            'e_dc_drawn_mo_t': e_dc_drawn_mo_t,
            'e_dc_drawn_hfi_t': e_dc_drawn_hfi_t,
            'e_dc_drawn_t': e_dc_drawn_t,
            'e_dumped_heat_t': e_dumped_heat_t,
            'e_gen_heat_t': e_gen_heat_t,
            'e_load_heat_t': e_load_heat_t,
            'e_res_heat_t': e_res_heat_t,
            'p_in_avg': p_in_avg,
            'p_load_avg': p_load_avg,
            'p_waste_avg': p_waste_avg,
            'p_mo_heat_avg': p_mo_heat_avg,
            'p_hfi_heat_avg': p_hfi_heat_avg,
            'p_res_heat_avg': p_res_heat_avg,
            'p_dumped_heat_avg': p_dumped_heat_avg,
            'p_gen_heat_avg': p_gen_heat_avg,
            'last_time_data': last_time_data
        }

    # ═══════════════════════════════════════════════════════════════════════
    # CALCULATE & PLOT  (unchanged from V34)
    # ═══════════════════════════════════════════════════════════════════════

    def calculate_and_plot(self):
        f_base  = float(self.v_f.get()) * 1e6
        N_harm  = int(self.v_n.get())

        R_load    = float(self.v_rload.get()) + 1e-12
        C_load_val = float(self.v_cload.get())
        L_TL_val  = float(self.v_ltl.get()) * 1e-9
        R_mo      = float(self.v_rmo.get()) + 1e-12
        L_ch      = float(self.v_lch.get()) * 1e-9 + 1e-21
        R_hfi     = float(self.v_rhfi.get()) + 1e-12
        C_hi_val  = float(self.v_chi.get())
        R_gnd     = self._get_rgnd()

        mo_harms  = self.parse_mo_harms()

        active_resonators = []
        if self.v_global_res_enable.get():
            for res in self.resonators:
                if res['enabled'].get():
                    try:
                        active_resonators.append((
                            float(res['r1'].get()), float(res['c1'].get()),
                            float(res['l'].get()),  float(res['k'].get()),
                            float(res['rdc'].get()), float(res['cs'].get()),
                            float(res['cp'].get())))
                    except ValueError:
                        pass

        if not self.custom_loaded:
            delay = float(self.v_delay.get())
            ramp  = float(self.v_ramp.get())
            up    = float(self.v_up.get())
            vmin  = float(self.v_vmin.get())
            vmax  = float(self.v_vmax.get())
            mags, phases = self.generate_trapezoid(
                f_base, N_harm, delay, ramp, up, vmin, vmax)
        else:
            mags   = self.target_mags[:N_harm+1]
            phases = self.target_phases[:N_harm+1]
            if len(mags) < N_harm+1:
                mags   = np.pad(mags,   (0, N_harm+1-len(mags)))
                phases = np.pad(phases, (0, N_harm+1-len(phases)))

        V_node_req = mags * np.exp(1j*phases)

        if N_harm == 1:
            vmin_val = float(self.v_vmin.get())
            vmax_val = float(self.v_vmax.get())
            V_node_req[0] = ((vmax_val + vmin_val) / 2.0) + 0j
            if len(V_node_req) > 1:
                V_node_req[1] = ((vmax_val - vmin_val) / 2.0) * \
                                np.exp(1j*np.angle(V_node_req[1]))

        self.freqs = np.arange(N_harm+1) * f_base
        w = 2 * np.pi * self.freqs

        res_base = self._sim_core(
            [], f_base, N_harm, R_load, C_load_val, L_TL_val,
            R_mo, L_ch, R_hfi, C_hi_val, mo_harms, V_node_req, w, R_gnd=R_gnd)
        res_act = self._sim_core(
            active_resonators, f_base, N_harm, R_load, C_load_val, L_TL_val,
            R_mo, L_ch, R_hfi, C_hi_val, mo_harms, V_node_req, w, R_gnd=R_gnd)

        # No-predistortion run: generator outputs V_node_req directly (no boosting).
        # Only computed when checkbox is ticked to avoid slowing down normal operation.
        res_nopred = None
        if self.v_predist_off.get():
            res_nopred = self._sim_core(
                active_resonators, f_base, N_harm, R_load, C_load_val, L_TL_val,
                R_mo, L_ch, R_hfi, C_hi_val, mo_harms, V_node_req, w,
                predistort=False, R_gnd=R_gnd)

        self.V_mo_gen        = res_act['V_mo_gen']
        self.V_hfi_gen       = res_act['V_hfi_gen']
        self.V_mo_gen_base   = res_base['V_mo_gen']   # no-resonator baseline
        self.V_hfi_gen_base  = res_base['V_hfi_gen']
        self.last_time_data  = res_act['last_time_data']
        self.last_V_node_req = V_node_req.copy()
        pin_base = res_base['p_in_avg']   # same circuit, no resonators
        pin_act  = res_act['p_in_avg']

        # CV² benchmarks
        C_load_F   = float(self.v_cload.get()) * 1e-12
        C_hi_F     = float(self.v_chi.get())   * 1e-12
        C_total_F  = C_load_F + C_hi_F
        V_max      = float(np.max(np.abs(res_act['v_rec_node_t'])))
        e_half_cv2 = 0.5 * C_total_F * V_max**2 * 1e12   # pJ
        e_cv2      =       C_total_F * V_max**2 * 1e12   # pJ

        # FOM = fractional reduction in total generator draw due to resonators.
        # Both runs use the identical circuit (same C_hi, R_hfi, sources) so the
        # reference is always on the same power scale and cannot be gamed by C_hi.
        fom = (1.0 - pin_act / pin_base) if pin_base > 0 else 0.0

        self.last_metrics = {
            'p_in_total_W':      pin_act,
            'p_load_W':          res_act['p_load_avg'],
            'p_mo_heat_W':       res_act['p_mo_heat_avg'],
            'p_hfi_heat_W':      res_act['p_hfi_heat_avg'],
            'p_res_heat_W':      res_act['p_res_heat_avg'],
            'p_dump_heat_W':     res_act['p_dumped_heat_avg'],
            'p_gen_heat_W':      res_act['p_gen_heat_avg'],
            'p_waste_W':         res_act['p_waste_avg'],
            'fom':               fom,
            'p_in_baseline_W':   pin_base,
            'f_hz':              float(self.v_f.get()) * 1e6,
            'e_half_cv2_pJ':     e_half_cv2,
            'e_cv2_pJ':          e_cv2,
            'c_total_pF':        C_total_F * 1e12,
            'c_load_pF':         C_load_F  * 1e12,
            'c_hi_pF':           C_hi_F    * 1e12,
        }

        t_plot       = res_act['t_plot']
        v_targ_t     = res_act['v_targ_t']
        v_rec_node_t = res_act['v_rec_node_t']
        v_mo_t       = res_act['v_mo_t']
        v_hfi_t      = res_act['v_hfi_t']
        i_mo_t       = res_act['i_mo_t']
        i_hfi_t      = res_act['i_hfi_t']
        e_dc_drawn_t     = res_act['e_dc_drawn_t']
        e_dc_drawn_mo_t  = res_act['e_dc_drawn_mo_t']
        e_dc_drawn_hfi_t = res_act['e_dc_drawn_hfi_t']
        e_dumped_heat_t  = res_act['e_dumped_heat_t']
        e_gen_heat_t     = res_act['e_gen_heat_t']
        e_load_heat_t    = res_act['e_load_heat_t']
        e_res_heat_t     = res_act['e_res_heat_t']

        # ── Time Domain plots ────────────────────────────────────────────
        np_sfx  = ' (predist ON)'   if res_nopred else ''
        np_sfx2 = ' (predist OFF)'  if res_nopred else ''

        # subplot 1: Load voltage
        self.ax_time_load.clear()
        self.ax_time_load.plot(t_plot*1e9, v_targ_t, 'k', lw=4, alpha=0.3,
                               label='Target')
        self.ax_time_load.plot(t_plot*1e9, v_rec_node_t, 'r--', lw=1.5,
                               label=f'Load Voltage{np_sfx}')
        if res_nopred:
            self.ax_time_load.plot(t_plot*1e9, res_nopred['v_rec_node_t'],
                                   'g-', lw=1.5, label=f'Load Voltage{np_sfx2}')
        self.ax_time_load.set(title="Time Domain: Voltage at Load",
                              xlabel="Time (ns)", ylabel="Voltage (V)")
        self.ax_time_load.legend(fontsize='small')
        self.ax_time_load.grid(True)

        # subplot 2: Internal generator output
        self.ax_time_gen.clear()
        self.ax_time_gen2.clear()
        line1 = self.ax_time_gen.plot(t_plot*1e9, v_mo_t, 'b',
                                      label=f'V_MO{np_sfx}')
        if res_nopred:
            self.ax_time_gen.plot(t_plot*1e9, res_nopred['v_mo_t'],
                                  color='green', lw=1.5, linestyle='--',
                                  label=f'V_MO{np_sfx2}')
        self.ax_time_gen.set_ylabel("V_MO (V)", color='blue')
        self.ax_time_gen.tick_params(axis='y', labelcolor='blue')
        if C_hi_val > 0:
            line2 = self.ax_time_gen2.plot(t_plot*1e9, v_hfi_t, 'tab:orange',
                                            label=f'V_HFI{np_sfx}')
            if res_nopred:
                self.ax_time_gen2.plot(t_plot*1e9, res_nopred['v_hfi_t'],
                                       color='olive', lw=1.5, linestyle='--',
                                       label=f'V_HFI{np_sfx2}')
            self.ax_time_gen2.set_ylabel("V_HFI (V)", color='tab:orange')
            self.ax_time_gen2.tick_params(axis='y', labelcolor='tab:orange')
            all_lines = (self.ax_time_gen.get_lines() +
                         self.ax_time_gen2.get_lines())
            self.ax_time_gen.legend(all_lines,
                                    [l.get_label() for l in all_lines],
                                    loc='upper left', fontsize='small')
        else:
            self.ax_time_gen.legend(loc='upper left', fontsize='small')
        self.ax_time_gen.set(title="Internal Generator Output", xlabel="Time (ns)")
        self.ax_time_gen.grid(True)
        # Sync x-limits on twin axis explicitly — twinx does not auto-update on clear()
        x_lo, x_hi = t_plot[0]*1e9, t_plot[-1]*1e9
        self.ax_time_gen.set_xlim(x_lo, x_hi)
        self.ax_time_gen2.set_xlim(x_lo, x_hi)

        # subplot 3: Energy breakdown
        self.ax_energy.clear()
        self.ax_energy.plot(t_plot*1e9, e_dc_drawn_mo_t*1e9,
                            color='blue', lw=1.5, label=f'MO DC Energy{np_sfx}')
        if C_hi_val > 0:
            self.ax_energy.plot(t_plot*1e9, e_dc_drawn_hfi_t*1e9,
                                color='tab:orange', lw=1.5, label=f'HFI DC Energy{np_sfx}')
        self.ax_energy.plot(t_plot*1e9, e_dc_drawn_t*1e9,
                            'k--', lw=1.5, label=f'Total DC Energy{np_sfx}')
        self.ax_energy.plot(t_plot*1e9, e_load_heat_t*1e9,
                            'g-', lw=2, label=f'R_Load loss (unavoidable){np_sfx}')
        if res_nopred:
            np_e = res_nopred
            self.ax_energy.plot(t_plot*1e9,
                                (np_e['e_dc_drawn_mo_t'] + np_e['e_dc_drawn_hfi_t'])*1e9,
                                color='gray', lw=1.5, linestyle='--',
                                label=f'Total DC Energy{np_sfx2}')
            self.ax_energy.plot(t_plot*1e9, np_e['e_load_heat_t']*1e9,
                                color='limegreen', lw=1.5, linestyle='--',
                                label=f'R_Load loss (unavoidable){np_sfx2}')
        self.ax_energy.plot(t_plot*1e9, e_dumped_heat_t*1e9,
                            'c-', lw=2, label='Burned Returned Energy (Dump)')
        self.ax_energy.plot(t_plot*1e9, e_gen_heat_t*1e9,
                            'r-', lw=2, label='Dissipated (Gen I²R)')
        if active_resonators:
            self.ax_energy.plot(t_plot*1e9, e_res_heat_t*1e9,
                                'm-', lw=2, label='Dissipated (Resonators)')
        self.ax_energy.set(title="System Energy Breakdown",
                           ylabel="Energy (nJ)", xlabel="Time (ns)")
        self.ax_energy.grid(True)
        self.ax_energy.legend(loc='upper left', fontsize='small')

        # subplot 4: Generator branch currents
        self.ax_current.clear()
        self.ax_current.plot(t_plot*1e9, i_mo_t*1000, 'b',
                             label=f'I_MO{np_sfx}')
        if res_nopred:
            self.ax_current.plot(t_plot*1e9, res_nopred['i_mo_t']*1000,
                                 'g--', lw=1.5, label=f'I_MO{np_sfx2}')
        if C_hi_val > 0:
            self.ax_current.plot(t_plot*1e9, i_hfi_t*1000, 'tab:orange',
                                 label=f'I_HFI{np_sfx}')
            if res_nopred:
                self.ax_current.plot(t_plot*1e9, res_nopred['i_hfi_t']*1000,
                                     color='olive', lw=1.5, linestyle='--',
                                     label=f'I_HFI{np_sfx2}')
        self.ax_current.set(title="Generator Branch Currents",
                            xlabel="Time (ns)", ylabel="Current (mA)")
        self.ax_current.grid(True)
        self.ax_current.legend(loc='upper right', fontsize='small')

        self.canvas_time.draw()
        # ── Frequency Domain plots ───────────────────────────────────────
        f_base_mhz  = f_base / 1e6
        max_freq_mhz = N_harm * f_base_mhz
        margin       = f_base_mhz * 0.5
        V_gen_total  = self.V_mo_gen + self.V_hfi_gen
        V_gen_base   = res_base['V_mo_gen'] + res_base['V_hfi_gen']

        # AC harmonics only — DC dwarfs everything else
        ac_idx   = np.arange(1, N_harm + 1)
        freqs_ac = self.freqs[ac_idx] / 1e6
        mag_targ = np.abs(V_node_req[ac_idx])
        mag_base = np.abs(V_gen_base[ac_idx])
        mag_act  = np.abs(V_gen_total[ac_idx])

        self.ax_freq_mag.clear()
        w_bar = 0.25 * f_base_mhz   # bar half-width
        self.ax_freq_mag.bar(freqs_ac - w_bar, mag_targ, width=w_bar*1.8,
                             color='black', alpha=0.5, label='Target Node')
        self.ax_freq_mag.bar(freqs_ac,          mag_base, width=w_bar*1.8,
                             color='steelblue', alpha=0.7, label='No resonators')
        self.ax_freq_mag.bar(freqs_ac + w_bar,  mag_act,  width=w_bar*1.8,
                             color='tomato',    alpha=0.8, label='With resonators')
        self.ax_freq_mag.set(title="Generator Spectrum Magnitude (AC, DC excluded)",
                             yscale='log', xlabel="Freq (MHz)", ylabel="|V_gen| (V)",
                             xlim=(-margin, max_freq_mhz + margin))
        self.ax_freq_mag.legend(fontsize='small')
        self.ax_freq_mag.grid(True, alpha=0.3, axis='y')

        self.ax_freq_phase.clear()
        self.ax_freq_phase.plot(self.freqs[1:]/1e6,
                                np.degrees(np.angle(V_node_req)[1:]),
                                'ko', alpha=0.5, label='Target Node')
        self.ax_freq_phase.plot(self.freqs[1:]/1e6,
                                np.degrees(np.angle(V_gen_total)[1:]),
                                'rx', label='Predistorted Total')
        self.ax_freq_phase.set(title="Generator Spectrum Phase",
                               xlabel="Freq (MHz)", ylabel="Phase (°)",
                               xlim=(f_base_mhz - margin, max_freq_mhz + margin))
        self.ax_freq_phase.legend()
        self.ax_freq_phase.grid(True)

        # Metrics labels
        def _np_sfx(key):
            if res_nopred is None:
                return ''
            return f'  [no-predist: {res_nopred[key]:.3e} W]'

        self.lbl_p_in_total.config(
            text=f"Total DC Power Drawn: {res_act['p_in_avg']:.3e} W"
                 + _np_sfx('p_in_avg'))
        self.lbl_p_baseline.config(
            text=f"  Baseline (no resonators): {pin_base:.3e} W")
        self.lbl_p_load.config(
            text=(f"R_Load loss (unavoidable): {res_act['p_load_avg']:.3e} W"
                  + _np_sfx('p_load_avg')
                  + f"  |  R_MO loss: {res_act['p_mo_heat_avg']:.3e} W"
                  + f"  |  R_HFI loss: {res_act['p_hfi_heat_avg']:.3e} W"))
        self.lbl_p_res.config(
            text=f"Resonator heat (R1+R_DC+R_var): {res_act['p_res_heat_avg']:.3e} W")
        self.lbl_p_waste.config(
            text=f"Cap. discharge returned to gen (not recycled): {res_act['p_dumped_heat_avg']:.3e} W")

        # Delivery efficiency: fraction of drawn energy that reaches R_load
        # R_MO efficiency: fraction of drawn energy that is NOT burned in R_MO
        p_drawn = res_act['p_in_avg']
        eta_del = res_act['p_load_avg'] / p_drawn if p_drawn > 0 else 0.0
        eta_mo  = 1.0 - res_act['p_mo_heat_avg'] / p_drawn if p_drawn > 0 else 0.0

        # Cp burden: what fraction of the total parallel capacitance (C_par = Cp+Cload+Chi)
        # is the parasitic Cp? At anti-resonance the resonator cancels all C_par equally,
        # but only Cload does useful work. Cp/C_par is the "wasted recycling fraction".
        total_Cp_pF = sum(
            float(res['cp'].get())
            for res in self.resonators
            if res['enabled'].get() and self.v_global_res_enable.get()
            and float(res['cp'].get()) > 0
        ) if self.resonators else 0.0
        C_load_pF = float(self.v_cload.get())
        C_hi_pF   = float(self.v_chi.get())
        C_par_pF  = total_Cp_pF + C_load_pF + C_hi_pF
        cp_burden = total_Cp_pF / C_par_pF if C_par_pF > 0 else 0.0

        cp_str = (f"   |   Cp burden = Cp/C_par: {cp_burden:.3f}"
                  f" ({total_Cp_pF:.2g}/{C_par_pF:.4g} pF)"
                  if total_Cp_pF > 0 else "")
        self.lbl_eta_del.config(
            text=f"Delivery eff. η_del = E_Rload/E_drawn: {eta_del:.3f}"
                 f"   |   R_MO eff. η_MO = 1-E_Rmo/E_drawn: {eta_mo:.3f}"
                 + cp_str)

        fom_note = ("  ⚠ resonator costs more than it saves" if fom < 0 else "")
        e_halfcv2_J = 0.5 * C_total_F * V_max**2
        f_hz_now    = float(self.v_f.get()) * 1e6
        overhead_N  = (pin_act / f_hz_now) / e_halfcv2_J if e_halfcv2_J > 0 else 0.0
        # FOM/N = 1/N_act - 1/N_base = energy saved by resonators in units of ½CV² per cycle.
        # Rewards both high recycling AND low absolute draw. Unaffected by R_MO tricks.
        N_base      = (pin_base / f_hz_now) / e_halfcv2_J if e_halfcv2_J > 0 else 1.0
        fom_over_n  = (1.0/overhead_N - 1.0/N_base) if overhead_N > 0 and N_base > 0 else 0.0
        self.lbl_fom.config(
            text=f"Recycl. FOM: {fom:.3f}{fom_note}"
                 f"   |   Overhead N: {overhead_N:.1f}×½CV²"
                 f"   |   FOM/N: {fom_over_n:.4f}")

        # ── Energy per cycle = Power / f_base ────────────────────────────
        f_hz  = float(self.v_f.get()) * 1e6
        T_cyc = 1.0 / f_hz  if f_hz > 0 else 1.0

        def pJ(p): return p * T_cyc * 1e12   # W → pJ per cycle
        def _np_pJ(key):
            if res_nopred is None:
                return ''
            return f'  [no-predist: {pJ(res_nopred[key]):.3f} pJ]'

        self.lbl_e_in.config(
            text=f"  E_drawn (from MO+HFI):               {pJ(res_act['p_in_avg']):.3f} pJ"
                 + _np_pJ('p_in_avg'))
        self.lbl_e_load.config(
            text=f"  E_Rload (unavoidable I²R in R_Load):  {pJ(res_act['p_load_avg']):.3f} pJ"
                 + _np_pJ('p_load_avg'))
        self.lbl_e_rmo.config(
            text=f"  E_R_MO  (I²R loss, MO branch):       {pJ(res_act['p_mo_heat_avg']):.3f} pJ"
                 + _np_pJ('p_mo_heat_avg'))
        self.lbl_e_res.config(
            text=f"  E_res   (loss in resonators):        {pJ(res_act['p_res_heat_avg']):.3f} pJ")
        self.lbl_e_dump.config(
            text=f"  E_ret   (cap discharge, not recycled): {pJ(res_act['p_dumped_heat_avg']):.3f} pJ")

        # ½CV²_max and CV²_max benchmarks — already computed above
        c_load_pF  = C_load_F * 1e12
        c_hi_pF    = C_hi_F   * 1e12
        c_total_pF = C_total_F * 1e12
        cv2_label  = (f"C_load+C_hi={c_load_pF:.4g}+{c_hi_pF:.4g}={c_total_pF:.4g}pF"
                      if c_hi_pF > 0 else f"C_load={c_load_pF:.4g}pF")
        self.lbl_e_cv2.config(
            text=f"  ½CV²_max ({cv2_label}, adiabatic limit): {e_half_cv2:.3f} pJ")
        self.lbl_e_cv2full.config(
            text=f"  CV²_max  ({cv2_label}, fast-switch limit): {e_cv2:.3f} pJ")

        self.run_ac_sweep()

    # ═══════════════════════════════════════════════════════════════════════
    # SPECTRA  — generator spectrum with all cases shown
    # ═══════════════════════════════════════════════════════════════════════

    def calculate_spectra(self):
        """
        Compute and display generator spectra for 4 cases:
          1. Target trapezoid (black)
          2. Predistorted — no resonators (blue)
          3. Predistorted — with enabled resonators (red)
          4. Predistorted — MO only (HFI disabled), if C_hi > 0 (orange)
        All AC harmonics shown (DC excluded). Phase plot gets same series.
        """
        try:
            f_base    = float(self.v_f.get()) * 1e6
            N_harm    = int(self.v_n.get())
            R_load    = float(self.v_rload.get()) + 1e-12
            C_load_val = float(self.v_cload.get())
            L_TL_val  = float(self.v_ltl.get()) * 1e-9
            R_mo      = float(self.v_rmo.get()) + 1e-12
            L_ch      = float(self.v_lch.get()) * 1e-9 + 1e-21
            R_hfi     = float(self.v_rhfi.get()) + 1e-12
            C_hi_val  = float(self.v_chi.get())
            R_gnd     = self._get_rgnd()
            mo_harms  = self.parse_mo_harms()
        except Exception as e:
            messagebox.showerror("Error", f"Could not read circuit params: {e}")
            return

        active_resonators = []
        if self.v_global_res_enable.get():
            for res in self.resonators:
                if res['enabled'].get():
                    try:
                        active_resonators.append((
                            float(res['r1'].get()), float(res['c1'].get()),
                            float(res['l'].get()),  float(res['k'].get()),
                            float(res['rdc'].get()), float(res['cs'].get()),
                            float(res['cp'].get())))
                    except ValueError:
                        pass

        if not self.custom_loaded:
            mags, phases = self.generate_trapezoid(
                f_base, N_harm,
                float(self.v_delay.get()), float(self.v_ramp.get()),
                float(self.v_up.get()), float(self.v_vmin.get()), float(self.v_vmax.get()))
        else:
            mags   = self.target_mags[:N_harm+1]
            phases = self.target_phases[:N_harm+1]
            if len(mags) < N_harm+1:
                mags   = np.pad(mags,   (0, N_harm+1-len(mags)))
                phases = np.pad(phases, (0, N_harm+1-len(phases)))

        V_node_req = mags * np.exp(1j*phases)
        w = 2 * np.pi * np.arange(N_harm+1) * f_base
        self.freqs = np.arange(N_harm+1) * f_base

        # Run 3 sims
        res_base = self._sim_core(
            [], f_base, N_harm, R_load, C_load_val, L_TL_val,
            R_mo, L_ch, R_hfi, C_hi_val, mo_harms, V_node_req, w, R_gnd=R_gnd)
        res_act = self._sim_core(
            active_resonators, f_base, N_harm, R_load, C_load_val, L_TL_val,
            R_mo, L_ch, R_hfi, C_hi_val, mo_harms, V_node_req, w, R_gnd=R_gnd)
        # 4th case (orange): MO only with HFI disabled — only shown when HFI is active (C_hi > 0).
        # When HFI is off (C_hi=0), this sim is skipped and no orange bar is shown.
        res_mo_only = None
        if C_hi_val > 0:
            res_mo_only = self._sim_core(
                active_resonators, f_base, N_harm, R_load, C_load_val, L_TL_val,
                R_mo, L_ch, R_hfi, 0.0, mo_harms, V_node_req, w, R_gnd=R_gnd)

        # Store for save_spectra
        self.V_mo_gen       = res_act['V_mo_gen']
        self.V_hfi_gen      = res_act['V_hfi_gen']
        self.V_mo_gen_base  = res_base['V_mo_gen']
        self.V_hfi_gen_base = res_base['V_hfi_gen']
        self.last_V_node_req = V_node_req.copy()

        V_targ    = V_node_req
        V_no_res  = res_base['V_mo_gen'] + res_base['V_hfi_gen']
        V_with    = res_act['V_mo_gen']  + res_act['V_hfi_gen']
        V_mo_only = (res_mo_only['V_mo_gen'] + res_mo_only['V_hfi_gen']
                     if res_mo_only else None)

        ac        = np.arange(1, N_harm+1)
        freqs_ac  = self.freqs[ac] / 1e6
        f_base_mhz = f_base / 1e6
        max_f      = N_harm * f_base_mhz
        margin     = f_base_mhz * 0.5

        # ── Magnitude plot ───────────────────────────────────────────────
        self.ax_freq_mag.clear()
        n_groups = 4 if V_mo_only is not None else 3
        w_bar  = 0.8 * f_base_mhz / n_groups
        offsets = np.linspace(-(n_groups-1)/2, (n_groups-1)/2, n_groups) * w_bar

        self.ax_freq_mag.bar(freqs_ac + offsets[0], np.abs(V_targ[ac]),
                             width=w_bar*0.9, color='black',     alpha=0.55, label='Target trapezoid')
        self.ax_freq_mag.bar(freqs_ac + offsets[1], np.abs(V_no_res[ac]),
                             width=w_bar*0.9, color='steelblue', alpha=0.8,  label='Predist. no resonators')
        self.ax_freq_mag.bar(freqs_ac + offsets[2], np.abs(V_with[ac]),
                             width=w_bar*0.9, color='tomato',    alpha=0.85, label='Predist. with resonators')
        if V_mo_only is not None:
            self.ax_freq_mag.bar(freqs_ac + offsets[3], np.abs(V_mo_only[ac]),
                                 width=w_bar*0.9, color='darkorange', alpha=0.8,
                                 label='Predist. MO only (HFI off)')
        self.ax_freq_mag.set(title="Generator Spectrum Magnitude (AC, DC excluded)",
                             yscale='log', xlabel="Freq (MHz)", ylabel="|V_gen| (V)",
                             xlim=(-margin, max_f + margin))
        self.ax_freq_mag.legend(fontsize='small')
        self.ax_freq_mag.grid(True, alpha=0.3, axis='y')

        # ── Phase plot ───────────────────────────────────────────────────
        self.ax_freq_phase.clear()
        self.ax_freq_phase.plot(freqs_ac, np.degrees(np.angle(V_targ[ac])),
                                'ko', ms=5, alpha=0.5, label='Target')
        self.ax_freq_phase.plot(freqs_ac, np.degrees(np.angle(V_no_res[ac])),
                                'b^', ms=5, label='No resonators')
        self.ax_freq_phase.plot(freqs_ac, np.degrees(np.angle(V_with[ac])),
                                'rs', ms=5, label='With resonators')
        if V_mo_only is not None:
            self.ax_freq_phase.plot(freqs_ac, np.degrees(np.angle(V_mo_only[ac])),
                                    'd', color='darkorange', ms=5, label='MO only')
        self.ax_freq_phase.set(title="Generator Spectrum Phase",
                               xlabel="Freq (MHz)", ylabel="Phase (°)",
                               xlim=(f_base_mhz - margin, max_f + margin))
        self.ax_freq_phase.legend(fontsize='small')
        self.ax_freq_phase.grid(True)

        self.fig_freq.tight_layout(pad=2.5)
        self.canvas_freq.draw()

        # Switch to Frequency Domain tab
        freq_tab_idx = list(self.plot_notebook.tabs()).index(str(self.tab_freq))
        self.plot_notebook.select(freq_tab_idx)

    # ═══════════════════════════════════════════════════════════════════════
    # AC SWEEP  (unchanged from V34)
    # ═══════════════════════════════════════════════════════════════════════

    def run_ac_sweep(self):
        f_start = float(self.v_f_start.get()) * 1e6
        f_stop  = float(self.v_f_stop.get())  * 1e6
        pts     = int(self.v_f_pts.get())
        freqs_sweep = np.linspace(f_start, f_stop, pts)
        omega       = 2 * np.pi * freqs_sweep

        f_base_mhz = float(self.v_f.get())
        N_harm     = int(self.v_n.get())

        R_load    = float(self.v_rload.get()) + 1e-12
        R_mo      = float(self.v_rmo.get())   + 1e-12
        L_ch      = float(self.v_lch.get())   * 1e-9 + 1e-21
        R_hfi     = float(self.v_rhfi.get())  + 1e-12
        C_load_val = float(self.v_cload.get())
        C_hi_val  = float(self.v_chi.get())
        L_TL_val  = float(self.v_ltl.get())   * 1e-9
        R_gnd     = self._get_rgnd()

        Z_load_base = R_load + 1j*omega*L_TL_val
        if C_load_val > 0:
            ac_mask = (omega != 0)
            Z_lb_ac = Z_load_base.copy()
            Z_lb_ac[ac_mask] += 1 / (1j*omega[ac_mask]*(C_load_val*1e-12))
            if not ac_mask.all():
                Z_lb_ac[~ac_mask] = np.inf
            Z_load_base = Z_lb_ac

        Y_load_base = 1 / Z_load_base
        # MO Thevenin: R_toGND at junction X between R_MO and L_choke
        R_gnd_safe  = max(R_gnd, 1.0)
        R_par_mo    = R_mo * R_gnd_safe / (R_mo + R_gnd_safe)   # R_mo || R_toGND
        Y_mo        = 1 / (R_par_mo + 1j*omega*L_ch)
        Y_hfi = (1.0 / (R_hfi + 1.0/(1j*omega*(C_hi_val*1e-12)))
                 if C_hi_val > 0
                 else np.zeros_like(omega, dtype=complex))

        active_resonators = []
        if self.v_global_res_enable.get():
            for res in self.resonators:
                if res['enabled'].get():
                    try:
                        active_resonators.append((
                            float(res['r1'].get()), float(res['c1'].get()),
                            float(res['l'].get()),  float(res['k'].get()),
                            float(res['rdc'].get()), float(res['cs'].get()),
                            float(res['cp'].get())))
                    except ValueError:
                        pass

        Y_res_total = np.zeros_like(omega, dtype=complex)
        for (R1, C1, L, k_val, R_DC, C_s, C_p) in active_resonators:
            Z_Ls_arr    = self.get_Ls_impedance_array(freqs_sweep, R_DC, R1, C1, L, k_val)
            Y_s         = (1 / (Z_Ls_arr + 1/(1j*omega*C_s))
                           if C_s > 0 else np.zeros_like(omega, dtype=complex))
            Y_res_total += (Y_s + 1j*omega*C_p)

        Y_tot = Y_load_base + Y_res_total + Y_hfi + Y_mo
        H_mo  = Y_mo  / Y_tot
        H_hfi = Y_hfi / Y_tot
        I_ch_sweep   = (1.0 - H_mo) * Y_mo
        I_load_sweep = H_mo * Y_load_base

        # No-resonator baseline transfer function
        Y_tot_bl = Y_load_base + Y_hfi + Y_mo
        H_mo_bl  = Y_mo / Y_tot_bl

        self.ax_vload_f.clear()
        self.ax_vload_f.plot(freqs_sweep/1e6, np.abs(H_mo_bl),
                             'b--', lw=1.0, alpha=0.6, label='MO Transfer — no resonators')
        self.ax_vload_f.plot(freqs_sweep/1e6, np.abs(H_mo),
                             'b-', label='MO Transfer — with resonators')
        if C_hi_val > 0:
            self.ax_vload_f.plot(freqs_sweep/1e6, np.abs(H_hfi),
                                 'r-', label='HFI Transfer')

        for n in range(1, N_harm+1):
            if (f_start/1e6) <= n*f_base_mhz <= (f_stop/1e6):
                self.ax_vload_f.axvline(x=n*f_base_mhz, color='gray',
                                        linestyle='--', alpha=0.5)

        self.ax_vload_f.set(
            title="Voltage Transfer |V_node/V_gen|  (dips = resonator loading generator)",
            yscale='log', xlabel="Freq (MHz)", ylabel="|H| (V/V)")
        self.ax_vload_f.legend(fontsize='small')
        self.ax_vload_f.grid(True, which="both", ls="--", alpha=0.5)

        self.ax_iratio_f.clear()
        self.ax_iratio_f.plot(
            freqs_sweep/1e6,
            np.clip(np.abs(I_load_sweep) / (np.abs(I_ch_sweep) + 1e-20),
                    1e-6, 1e6),
            'g-', label='|I_load / I_choke|')
        self.ax_iratio_f.axhline(y=1.0, color='k', linestyle=':', lw=0.8, alpha=0.6)
        for n in range(1, N_harm+1):
            if (f_start/1e6) <= n*f_base_mhz <= (f_stop/1e6):
                self.ax_iratio_f.axvline(x=n*f_base_mhz, color='gray',
                                         linestyle='--', alpha=0.5)
        self.ax_iratio_f.set(
            title="|I_load / I_choke| — MO Current Ratio",
            yscale='log', xlabel="Freq (MHz)", ylabel="Ratio")
        self.ax_iratio_f.legend()
        self.ax_iratio_f.grid(True, which="both", ls="--", alpha=0.5)
        self.canvas_freq.draw()

        # ── Find and display |I_load/I_ser| peak frequencies ─────────────
        # Baseline ratio without resonators = |Y_load_base| / |Y_load_base + Y_hfi|.
        # A peak is a local maximum above the baseline (not hardcoded 1.0, which
        # breaks when C_hi is large and the ratio never reaches 1).
        ratio_arr = np.clip(np.abs(I_load_sweep) / (np.abs(I_ch_sweep) + 1e-20),
                            1e-6, 1e6)

        # Compute baseline (no-resonator) at each sweep frequency for threshold
        Y_res_zero  = np.zeros_like(omega, dtype=complex)
        Y_tot_bl    = Y_load_base + Y_hfi + Y_mo
        I_ch_bl     = (1.0 - Y_mo / Y_tot_bl) * Y_mo
        I_load_bl   = (Y_mo / Y_tot_bl) * Y_load_base
        ratio_bl    = np.clip(np.abs(I_load_bl) / (np.abs(I_ch_bl) + 1e-20), 1e-6, 1e6)
        # Use the median baseline across the sweep as the threshold
        baseline_threshold = float(np.median(ratio_bl))

        peaks = []
        for i in range(1, len(ratio_arr) - 1):
            if ratio_arr[i] > ratio_arr[i-1] and ratio_arr[i] > ratio_arr[i+1]:
                if ratio_arr[i] > baseline_threshold:
                    peaks.append((ratio_arr[i], freqs_sweep[i]))
        peaks.sort(key=lambda x: -x[0])
        peaks = peaks[:N_harm]
        peaks.sort(key=lambda x: x[1])   # ascending frequency

        # Per-resonator tab: match each peak to its resonator by harm#.
        # Find the peak closest in frequency to harm# * f_base for each resonator.
        f_base_hz = float(self.v_f.get()) * 1e6
        try:
            cload_F = float(self.v_cload.get()) * 1e-12
        except Exception:
            cload_F = 50e-12
        try:
            chi_F = float(self.v_chi.get()) * 1e-12
        except Exception:
            chi_F = 0.0

        enabled_res = [res for res in self.resonators if res['enabled'].get()]
        for res in enabled_res:
            try:
                l  = float(res['l'].get())
                cs = float(res['cs'].get())
                f_ser = 1.0 / (2.0 * np.pi * np.sqrt(l * cs)) if l > 0 and cs > 0 else 0.0
                ser_str = f"{f_ser/1e6:.3f} MHz" if f_ser > 0 else "N/A"
            except Exception:
                ser_str = "N/A"

            try:
                harm_n = int(res['harm'].get())
            except Exception:
                harm_n = 1
            f_target_res = harm_n * f_base_hz
            if peaks:
                closest = min(peaks, key=lambda p: abs(p[1] - f_target_res))
                ratio_val, f_peak = closest
                mult = f_peak / f_base_hz if f_base_hz > 0 else 0.0
                peak_str = f"{f_peak/1e6:.3f} MHz  ratio={ratio_val:.3f}  (×{mult:.2f}f₀)"
            else:
                peak_str = "--"

            res['lbl'].config(
                text=f"Harm {harm_n} | Series F_res: {ser_str}  |  Peak: {peak_str}")

    # ═══════════════════════════════════════════════════════════════════════
    # FILE I/O
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _timestamp():
        """Return a compact timestamp string for prefixing filenames."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    @staticmethod
    def _ts_path(directory, stem, ext):
        """Build a timestamped filepath: <dir>/<YYYYMMDD_HHMMSS>_<stem>.<ext>"""
        return os.path.join(directory, f"{WaveformSimulatorApp._timestamp()}_{stem}.{ext}")

    # ── Spectra ──────────────────────────────────────────────────────────

    def save_spectra(self):
        """Save Generator Spectra CSV + Res_Info CSV with same timestamp."""
        if self.V_mo_gen is None or self.last_V_node_req is None:
            messagebox.showwarning("No Data", "Run Calculate Spectra first.")
            return
        ts   = self._timestamp()
        init = f"{ts}_Generator_Spectra.csv"
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv", initialfile=init,
            filetypes=[("CSV", "*.csv"), ("All Files", "*.*")])
        if not filepath:
            return
        try:
            # ── Spectra CSV ──────────────────────────────────────────────
            V_orig   = self.last_V_node_req
            V_pred   = self.V_mo_gen + self.V_hfi_gen
            has_base = (self.V_mo_gen_base is not None and
                        self.V_hfi_gen_base is not None)
            V_base   = (self.V_mo_gen_base + self.V_hfi_gen_base
                        if has_base else None)
            N = len(V_orig)
            freqs_mhz = self.freqs / 1e6 if self.freqs is not None else np.arange(N)
            with open(filepath, 'w', newline='') as f:
                w = csv.writer(f)
                header = ["Harmonic_#", "Frequency_MHz",
                          "Target_Magnitude", "Target_Phase_deg",
                          "NoRes_Magnitude",  "NoRes_Phase_deg",
                          "WithRes_Magnitude","WithRes_Phase_deg"]
                if has_base:
                    header += ["Ratio_WithRes_vs_NoRes"]
                w.writerow(header)
                for i in range(1, N):
                    row = [i, f"{freqs_mhz[i]:.6g}",
                           f"{abs(V_orig[i]):.8g}",
                           f"{np.degrees(np.angle(V_orig[i])):.6g}",
                           f"{abs(V_base[i]):.8g}"  if has_base else "",
                           f"{np.degrees(np.angle(V_base[i])):.6g}" if has_base else "",
                           f"{abs(V_pred[i]):.8g}",
                           f"{np.degrees(np.angle(V_pred[i])):.6g}"]
                    if has_base and abs(V_base[i]) > 1e-30:
                        row.append(f"{abs(V_pred[i])/abs(V_base[i]):.6g}")
                    elif has_base:
                        row.append("")
                    w.writerow(row)

            # ── Res_Info CSV (same timestamp, same directory) ─────────────
            base_dir  = os.path.dirname(filepath)
            info_path = os.path.join(base_dir, f"{ts}_Res_Info.csv")
            with open(info_path, 'w', newline='') as f:
                w = csv.writer(f)

                # Circuit parameters header
                w.writerow(["# Circuit Parameters"])
                w.writerow(["Cload_pF",   self.v_cload.get()])
                w.writerow(["Rload_Ohm",  self.v_rload.get()])
                w.writerow(["R_MO_Ohm",   self.v_rmo.get()])
                w.writerow(["L_choke_nH", self.v_lch.get()])
                w.writerow(["R_toGND_Ohm",self.v_rgnd.get()])
                try:
                    c_hi = float(self.v_chi.get())
                except Exception:
                    c_hi = 0.0
                if c_hi > 0:
                    w.writerow(["C_hi_pF", self.v_chi.get()])
                    w.writerow(["R_HFI_Ohm", self.v_rhfi.get()])
                w.writerow(["f0_MHz",      self.v_f.get()])
                w.writerow(["N_harmonics", self.v_n.get()])
                w.writerow(["MO_harmonics",self.v_mo_harms.get()])
                w.writerow([])

                # Resonator table
                w.writerow(["# Enabled Resonators"])
                w.writerow(["Harm_#", "L_s (H)", "Cs (F)", "Cp (F)",
                            "R1 (Ohm)", "C1 (F)", "k", "R_DC (Ohm)"])
                for res in self.resonators:
                    if not res['enabled'].get():
                        continue
                    if not self.v_global_res_enable.get():
                        break
                    try:
                        w.writerow([
                            int(res['harm'].get()),
                            res['l'].get(),
                            res['cs'].get(),
                            res['cp'].get(),
                            res['r1'].get(),
                            res['c1'].get(),
                            res['k'].get(),
                            res['rdc'].get(),
                        ])
                    except Exception:
                        pass

            messagebox.showinfo("Saved",
                                f"Spectra saved to:\n{filepath}\n\n"
                                f"Resonator info saved to:\n{info_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {e}")

    # ── Time dependences ─────────────────────────────────────────────────

    def save_time_plot(self):
        ts   = self._timestamp()
        init = f"{ts}_Time_Domain_Plot.png"
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png", initialfile=init,
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("All Files", "*.*")])
        if filepath:
            try:
                self.fig_time.savefig(filepath, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Saved", "Plot saved to:\n" + filepath)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save plot: {e}")

    def save_time_data(self):
        """Save time-dependence CSV and an accompanying parameters text file."""
        if not self.last_time_data:
            messagebox.showwarning("Warning", "No data to save. Run Calculate & Plot first.")
            return
        ts   = self._timestamp()
        init = f"{ts}_Time_Dependences.csv"
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv", initialfile=init,
            filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if not filepath:
            return
        try:
            # ── CSV data file ────────────────────────────────────────────
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.last_time_data.keys())
                writer.writerows(zip(*self.last_time_data.values()))

            # ── Parameters text file (same timestamp, same directory) ────
            base_dir  = os.path.dirname(filepath)
            param_path = os.path.join(base_dir, f"{ts}_Parameters.txt")
            self._save_parameters_file(param_path, ts)

            messagebox.showinfo("Saved",
                "Time data saved to:\n" + filepath +
                "\n\nParameters saved to:\n" + param_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {e}")

    def _save_parameters_file(self, filepath, ts):
        """Write a human-readable parameters file for the current simulation state."""
        m = self.last_metrics
        lines = []
        lines.append(f"Waveform Predistortion Simulator — Parameters")
        lines.append(f"Saved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  (ts={ts})")
        lines.append("=" * 60)

        # ── Power Metrics (first, as requested) ─────────────────────────
        lines.append("")
        lines.append("POWER METRICS")
        lines.append("-" * 40)
        if m:
            fom_pct = m.get('fom', 0.0) * 100
            lines.append(f"  Total DC Power Drawn (with resonators): {m.get('p_in_total_W', 0):.4e} W")
            lines.append(f"  Total DC Power Drawn (baseline, no res): {m.get('p_in_baseline_W', 0):.4e} W")
            lines.append(f"  Recycl. FOM:                             {fom_pct:.3f} %")
            lines.append(f"  R_Load loss (unavoidable):               {m.get('p_load_W', 0):.4e} W")
            lines.append(f"  R_MO loss (MO delivery resistance):      {m.get('p_mo_heat_W', 0):.4e} W")
            lines.append(f"  R_HFI loss (HFI delivery resistance):    {m.get('p_hfi_heat_W', 0):.4e} W")
            lines.append(f"  Resonator heat (R1+R_DC+R_var):          {m.get('p_res_heat_W', 0):.4e} W")
            lines.append(f"  Dump heat (returned energy not recycled): {m.get('p_dump_heat_W', 0):.4e} W")
            lines.append(f"  Gen I2R heat (MO+HFI internal I2R):      {m.get('p_gen_heat_W', 0):.4e} W")
            # energy per cycle
            f_hz = m.get('f_hz', 33e6)
            T_c  = 1.0/f_hz if f_hz>0 else 1.0
            def pJ(p): return p*T_c*1e12
            lines.append("")
            lines.append("ENERGY PER CYCLE (pJ)")
            lines.append("-" * 40)
            lines.append(f"  E_drawn:     {pJ(m.get('p_in_total_W',0)):.3f} pJ")
            lines.append(f"  E_load:      {pJ(m.get('p_load_W',0)):.3f} pJ")
            lines.append(f"  E_R_MO:      {pJ(m.get('p_mo_heat_W',0)):.3f} pJ")
            lines.append(f"  E_res:       {pJ(m.get('p_res_heat_W',0)):.3f} pJ")
            lines.append(f"  E_dump:      {pJ(m.get('p_dump_heat_W',0)):.3f} pJ")
            c_lbl = (f"C_load+C_hi={m.get('c_load_pF',0):.4g}+{m.get('c_hi_pF',0):.4g}pF"
                     if m.get('c_hi_pF', 0) > 0 else f"C_load={m.get('c_load_pF',0):.4g}pF")
            lines.append(f"  1/2*CV2_max ({c_lbl}): {m.get('e_half_cv2_pJ',0):.3f} pJ  (adiabatic limit)")
            lines.append(f"  CV2_max     ({c_lbl}): {m.get('e_cv2_pJ',0):.3f} pJ  (fast-switch limit)")
        else:
            lines.append("  (not available — run Calculate & Plot first)")

        # ── R_gen / L_choke Optimizer Results ───────────────────────────
        opt = getattr(self, 'last_rmo_lch_opt', None)
        if opt:
            lines.append("")
            lines.append("R_gen / L_choke OPTIMIZER RESULTS")
            lines.append("-" * 40)
            lines.append(f"  R_gen constraint    >= {opt.get('R_min_constraint', 0):.1f} Ω")
            lines.append(f"  Optimal R_gen       = {opt['best_R_mo']:.4f} Ω")
            lines.append(f"  Optimal L_choke     = {opt['best_L_ch_nH']:.4f} nH")
            lines.append(f"  FOM at optimum      = {opt['fom_opt']:.4f}")
            lines.append(f"  Overhead N at opt.  = {opt['N_opt']:.3f} × ½CV²")
            lines.append(f"  FOM/N at optimum    = {opt['fomn_opt']:.5f}  ← maximised")
            lines.append(f"  E_drawn at optimum  = {opt['e_drawn_opt_pJ']:.3f} pJ/cycle")
            lines.append(f"  E_drawn (baseline)  = {opt['e_base_opt_pJ']:.3f} pJ/cycle")
            lines.append("")
            lines.append(f"  Current R_gen       = {opt['R_cur']:.4f} Ω")
            lines.append(f"  Current L_choke     = {opt['L_cur_nH']:.4f} nH")
            lines.append(f"  FOM/N current       = {opt['fomn_cur']:.5f}")
            lines.append(f"  E_drawn current     = {opt['e_drawn_cur_pJ']:.3f} pJ/cycle")
            lines.append(f"  Improvement (FOM/N) = {opt['improvement']:.2f}×")
        else:
            pass  # optimizer not yet run — omit section silently

        # ── Base Parameters ──────────────────────────────────────────────
        lines.append("")
        lines.append("BASE PARAMETERS")
        lines.append("-" * 40)
        lines.append(f"  Frequency:       {self.v_f.get()} MHz")
        lines.append(f"  N Harmonics:     {self.v_n.get()}")

        # ── Trapezoid Shape ──────────────────────────────────────────────
        lines.append("")
        lines.append("TRAPEZOID SHAPE")
        lines.append("-" * 40)
        lines.append(f"  Delay:           {self.v_delay.get()} x T")
        lines.append(f"  Ramp:            {self.v_ramp.get()} x T")
        lines.append(f"  Up Time:         {self.v_up.get()} x T")
        lines.append(f"  V_min:           {self.v_vmin.get()} V")
        lines.append(f"  V_max:           {self.v_vmax.get()} V")
        if self.custom_loaded:
            lines.append("  (Custom waveform loaded from CSV — trapezoid values not used)")

        # ── Circuit Components ───────────────────────────────────────────
        lines.append("")
        lines.append("CIRCUIT COMPONENTS")
        lines.append("-" * 40)
        lines.append(f"  R_load:          {self.v_rload.get()} Ω")
        lines.append(f"  C_load:          {self.v_cload.get()} pF")
        lines.append(f"  L_TL:            {self.v_ltl.get()} nH")
        lines.append(f"  R_MO:            {self.v_rmo.get()} Ω")
        lines.append(f"  L_choke:         {self.v_lch.get()} nH")
        lines.append(f"  R_HFI:           {self.v_rhfi.get()} Ω")
        lines.append(f"  R_toGND:         {self.v_rgnd.get()} Ω")
        lines.append(f"  C_hi:            {self.v_chi.get()} pF")

        # ── Generator Logic ──────────────────────────────────────────────
        lines.append("")
        lines.append("GENERATOR LOGIC")
        lines.append("-" * 40)
        lines.append(f"  MO Harmonics:    {self.v_mo_harms.get()}")

        # ── Resonators ───────────────────────────────────────────────────
        lines.append("")
        lines.append("RESONATORS")
        lines.append("-" * 40)
        lines.append(f"  Global Enable:   {self.v_global_res_enable.get()}")
        for i, res in enumerate(self.resonators):
            en = res['enabled'].get()
            lines.append(f"  Resonator {i+1}:  {'ENABLED' if en else 'disabled'}")
            if en:
                lines.append(f"    R1={res['r1'].get()} Ω  C1={res['c1'].get()} F  "
                             f"L={res['l'].get()} H  k={res['k'].get()}")
                lines.append(f"    R_DC={res['rdc'].get()} Ω  "
                             f"Cs={res['cs'].get()} F  Cp={res['cp'].get()} F")

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')

    # ── Predistorted waveform ────────────────────────────────────────────

    def load_csv(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("CSV Files", "*.csv")])
        if not filepath:
            return
        try:
            with open(filepath, 'r') as f:
                reader = csv.reader(f)
                self.target_mags   = np.array(
                    [float(x) for x in next(reader) if x.strip()])
                self.target_phases = np.radians(
                    np.array([0.0] + [float(x) for x in next(reader) if x.strip()]))
                self.v_n.set(str(len(self.target_mags) - 1))
                self.custom_loaded = True
                self.calculate_and_plot()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV: {e}")

    def clear_csv(self):
        self.custom_loaded = False
        self.calculate_and_plot()

    def save_csv(self):
        ts   = self._timestamp()
        init = f"{ts}_Predistorted.csv"
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv", initialfile=init,
            filetypes=[("CSV", "*.csv")])
        if not filepath:
            return
        try:
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                V_total = self.V_mo_gen + self.V_hfi_gen
                writer.writerow(np.abs(V_total))
                writer.writerow(np.degrees(np.angle(V_total))[1:])
            messagebox.showinfo("Saved", "Saved to:\n" + filepath)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app  = WaveformSimulatorApp(root)
    root.mainloop()
