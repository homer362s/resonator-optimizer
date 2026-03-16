# =============================================================================
# Res_Opt_v1.00.py  —  forked from Resonator_Optimizer_1_22.py
#
# CHANGELOG
# ---------
# v1.00 (initial versioned release)
#   FIX-1  _run_rser_optimization / _done:
#     Previously only wrote best_rser back to the GUI and discarded p_final
#     (the Cs values optimised for that Rser). The reported FOM was computed
#     with those Cs, making the result unreproducible from the GUI state.
#     Now calls _update_grid_cs(p_final) and sets _cs_optimized = True so
#     the GUI reflects the exact (Rser, Cs) pair that achieved the reported FOM.
#
#   FIX-5  Structured auto-versioned save filenames for analysis output.
#     Format — Frequency:     Spec_49MHz-820-680-330_v1.dat
#              Frequency ref:  Spec_49MHz-ref.dat
#              Steady state:   SS-BP3_16-49MHz-820-680-330_v1.dat
#              Steady ref:     SS-BP3_16-49MHz-ref.dat
#     Frequency uses only base freq + L values (no stimulus info).
#     Steady state prefixes with stimulus abbreviation (text initials) and
#     numeric tokens from the stimulus filename (e.g. Ben_phas_3_16 → BP3_16).
#     Version auto-increments (v1→v2→…); reference files have no version suffix.
#     VERSION bumped to 1.23; main panel label updated to "Ver. x.xx" format.
#
#   FIX-4  Reference Calculation mode added to main panel.
#     Checkbox "Reference Calculation (resonators OFF)" sets all Cs to 1e-21 F
#     (1e-9 pF) inside _get_resonator_params_from_gui — below the model's own
#     1e-16 F inactive threshold, so resonators return open-circuit (inf) with
#     no division-by-zero.  Cs Optimisation and Best Inductors are blocked while
#     the mode is active.
#
#   FIX-3  Steady-state: periods reduced 10 → 5; V_gen show/hide checkbox added.
#     T = 5/f0 in both _compute_fom and _run_steady_state_analysis.
#     New BooleanVar show_vgen_var (default True) controls whether V_gen is
#     plotted in the Voltage Overlay; checkbox appears in the steady-state toolbar.
#
#   FIX-6  Sub-harmonic resonator rejection in _cs_for_max_ratio and
#     greedy_cs_pretune.  If the computed anti-resonance frequency f_ar
#     (≈ sqrt((Cs+Cp)/(L·Cs·Cp))/2π) is below the fundamental f0, the
#     resonator would block sub-harmonically and slightly perturb the 1st
#     harmonic without helping at the target harmonic.  Such candidates are
#     now returned as inactive (Cs=1e-17) in both search paths.
#
#   FIX-2  _cs_for_max_ratio / _joint_optimise_cs — upper harmonics missed:
#     cs_min was floored at 0.01 pF.  For larger inductors at upper harmonics,
#     cs_series_res = 1/(L·ω²) falls below that floor, and the anti-resonance
#     optimal Cs (≈ Cp/(ω²LCp−1)) also lies in that sub-0.01 pF region, so
#     the grid search never saw it.  Additionally the simplified cs_series_res
#     ignores C1 inside Z_AB_parallel, so the actual series resonance of
#     Z_series_block differs from 1/(Lω²).
#     Fix: lower the floor to 0.001 pF and start the search from
#     0.1 × cs_series_res (one decade below the simplified estimate) so the
#     peak is found regardless of model inaccuracies.  Upper search limit
#     raised to 10 000 pF to maintain symmetry.  The "inactive" return guard
#     is lowered to match the model's own 1e-16 F threshold.
# =============================================================================

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pandas as pd
from datetime import datetime
# scipy not required — optimization implemented with numpy only
import os
import sys
import threading
import queue

VERSION = "1.29"

CHANGELOG = {
    "1.29": "Robust Fundamental input: _get_f_fundamental() wraps all .get() calls "
            "with try/except, resets to 33 MHz on invalid input instead of crashing. "
            "Large Fundamental entry uses tk.Entry (reliable font support).",
    "1.28": "Rser optimisation now tweaks Cs via coordinate descent (reseed=False) "
            "at each Rser candidate instead of holding Cs fixed. Preserves "
            "optimised Cs values and avoids full redistribution. Writes tweaked Cs "
            "back to panel and marks _cs_optimized=True on completion.",
    "1.27": "Main panel redesigned: window narrowed to 520px, Fundamental (MHz) "
            "moved below resonator table with font size 16, default frequency "
            "changed to 33 MHz.",
    "1.26": "SA bug-fixes: (1) seeding now evaluates all 50 candidates and picks "
            "the best, instead of stopping at the first FOM≥0. "
            "(2) Convergence window widened to 80 steps, tolerance tightened to 0.01% "
            "to prevent premature early-exit at ~30% FOM. "
            "(3) Default T0 raised from 5 to 20 (FOM% units) so SA can escape "
            "shallow local optima early in the search.",
    "1.0":  "Initial release: Resonator Optimizer with GA/Greedy search, "
            "Cs optimisation, steady-state and frequency analysis.",
    "1.01": "Cs scan ceiling raised from 200 pF to 1000 pF. "
            "Automatic revision tracking via CHANGELOG added.",
    "1.02": "Plot windows made separate Toplevels: steady-state, frequency, "
            "Cs-opt, GA optimiser each get their own resizable window.",
    "1.03": "Plot-specific controls moved into each plot window toolbar: "
            "freq range, V_node/S-param switch, subtract-ref checkbox, "
            "fundamental freq entry. Re-run button in each window.",
    "1.04": "Greedy search made monotonically non-decreasing: candidate locked "
            "only if FOM improves, otherwise position skipped.",
    "1.05": "Pruning criterion changed from Cs<0.1pF to current-ratio<min_ratio. "
            "Min ratio threshold configurable in Optimize menu (default 1.5x). "
            "Final inductor count now stable regardless of starting N. "
            "Display filter uses inactive sentinel (Cs<1e-16) not arbitrary Cs floor.",
    "1.25": "FIX-6: sub-harmonic resonator rejection in _cs_for_max_ratio and "
            "greedy_cs_pretune. Resonators whose anti-resonance frequency (approx. "
            "sqrt((Cs+Cp)/(L·Cs·Cp))/2π) falls below f0 are marked inactive.",
    "1.24": "Simulated Annealing added as 'SA' mode in Best Inductors window. "
            "Deterministic (seed=42), single-swap/double-swap/restart neighbour moves, "
            "Boltzmann acceptance, exponential cooling. Parameters: T0, α, Steps. "
            "Cs hard cap restored to 1000 pF.",
    "1.23": "Rser opt writes Cs back; Cs floor lowered for upper harmonics; "
            "Reference Calculation mode; structured auto-versioned filenames; "
            "V_gen checkbox; 5-period steady state; Ver. label on main panel.",
    "1.22": "Fixed Cs optimizer: searches for HIGH impedance (parallel anti-resonance). "
            "Cs search range now starts at cs_series_res = 1/(L*(2π*f)²) and goes up "
            "to 1000 pF so the L+Cs branch is inductive and anti-resonates with Cp. "
            "Useless components (best achievable ratio ≤ 1 at their target harmonic) "
            "are now marked inactive (Cs=1e-17) in both _cs_for_max_ratio and "
            "greedy_cs_pretune, preventing them from parking at 1000 pF and "
            "introducing spurious subharmonics into the Best Inductors search.",
}

class ResonatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Resonator Optimizer  v{VERSION}")
        self.root.geometry("520x800")
        _style = ttk.Style()
        _style.configure("Warn.TButton", foreground="red", font=("Helvetica", 9, "bold"))

        if getattr(sys, 'frozen', False):
            self.script_dir = os.path.dirname(sys.executable)
        else:
            self.script_dir = os.path.dirname(os.path.abspath(__file__))

        # --- Default Save Directory Setup ---
        self.default_save_dir = r"C:\Users\user\Documents\Resonator_data"
        try:
            if not os.path.exists(self.default_save_dir):
                os.makedirs(self.default_save_dir)
        except Exception as e:
            print(f"Could not create default dir {self.default_save_dir}: {e}")
            self.default_save_dir = self.script_dir

        # --- Inductor database state ---
        self.inductor_db = None          # DataFrame of loaded CSV
        self.inductor_db_path = tk.StringVar(value="")
        self.selected_inductors = []     # list of dicts from chosen rows

        # GA optimiser vars (needed by menu radiobuttons)
        self.ga_mode      = tk.StringVar(value="Greedy+GA")
        self.ga_n_res     = tk.IntVar(value=4)
        self.ga_pop       = tk.IntVar(value=30)
        self._pop_combobox = None   # set when optim window is built
        self.ga_gen       = tk.IntVar(value=40)
        self.ga_min_contrib = tk.DoubleVar(value=0.5)  # min ΔFOM% for leave-one-out keep
        self.ga_rser_min     = tk.DoubleVar(value=10.0)   # Ω
        self.ga_rser_max     = tk.DoubleVar(value=1000.0) # Ω
        # SA parameters
        self.sa_t0    = tk.DoubleVar(value=20.0)   # initial temperature (FOM % units)
        self.sa_alpha = tk.DoubleVar(value=0.97)  # cooling rate per step
        self.sa_iters = tk.IntVar(value=300)       # total SA steps

        # --- Menu Bar ---
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Files", menu=file_menu)
        file_menu.add_command(label="Open Stimulus",        command=self._menu_open_stimulus)
        file_menu.add_command(label="Open Resonators",      command=self._menu_open_resonators)
        file_menu.add_separator()
        file_menu.add_command(label="Save Analysis Data",    command=self._save_data)
        file_menu.add_command(label="Save Best Inductors Search", command=self._save_ga_progress)
        file_menu.add_command(label="Save Resonators",      command=self._save_resonators)
        file_menu.add_separator()
        file_menu.add_command(label="Open Inductor Database", command=self._menu_open_inductor_db)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._quit_app)

        self.inductor_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Choose Inductors", menu=self.inductor_menu)
        self.inductor_menu.add_command(label="(Load database first)", state="disabled")

        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Frequency Analysis",
            command=lambda: self._open_and_run("Frequency"))
        analysis_menu.add_command(label="Steady State",
            command=lambda: self._open_and_run("Steady State"))
        analysis_menu.add_separator()
        analysis_menu.add_command(label="Cs Optimisation",
            command=lambda: self._open_and_run("Cs Opt"))
        analysis_menu.add_command(label="Best Inductors",
            command=self._open_optim_window)

        # ── Fonts — match resonator table (small, system default) ──────────
        SF  = ("TkDefaultFont", 9)        # small — same as resonator grid
        SFB = ("TkDefaultFont", 9, "bold")

        # ── Single-column main panel ───────────────────────────────────────
        main_frame = ttk.Frame(self.root, padding="8")
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)   # resonator grid expands

        # ── Active Files ───────────────────────────────────────────────────
        file_frame = ttk.LabelFrame(main_frame, text="Active Files", padding="4")
        file_frame.grid(row=0, column=0, sticky="ew", pady=(0,4))
        file_frame.columnconfigure(1, weight=1)

        self.stimulus_file   = tk.StringVar(value="NHarm=4_Ben_phas 5_10.csv")
        self.resonators_file = tk.StringVar(value="Resonators-1.res")

        for row_i, (lbl, var, col) in enumerate([
                ("Stimulus:",    self.stimulus_file,   "blue"),
                ("Resonators:",  self.resonators_file, "blue"),
                ("Inductor DB:", self.inductor_db_path, "darkgreen")]):
            ttk.Label(file_frame, text=lbl, font=SF).grid(
                row=row_i, column=0, sticky="w", pady=1, padx=(0,4))
            ttk.Label(file_frame, textvariable=var, foreground=col,
                      font=SF, wraplength=280, justify="left").grid(
                row=row_i, column=1, sticky="w")

        # ── Circuit Parameters ─────────────────────────────────────────────
        params_frame = ttk.LabelFrame(main_frame, text="Circuit Parameters", padding="4")
        params_frame.grid(row=1, column=0, sticky="ew", pady=(0,4))

        self.V_gen_rms      = tk.DoubleVar(value=100.0)
        self.R_internal_gen = tk.DoubleVar(value=50.0)
        self.gen_matched_load = tk.BooleanVar(value=True)
        self.R_div1         = tk.DoubleVar(value=39.0)
        self.R_div2         = tk.DoubleVar(value=11.0)
        self.C_line         = tk.DoubleVar(value=0.0)
        self.Z_in_measure   = tk.DoubleVar(value=50.0)
        self.Rser           = tk.DoubleVar(value=100.0)
        self.Rload          = tk.DoubleVar(value=1.0)
        self.Cload          = tk.DoubleVar(value=50e-12)
        self.Lload          = tk.DoubleVar(value=3e-9)
        self.f_min          = tk.DoubleVar(value=20.0)
        self.f_max          = tk.DoubleVar(value=300.0)
        self.freq_plot_type  = tk.StringVar(value="V_node")
        self.s_param_choice  = tk.StringVar(value="Both")
        self.f_fundamental   = tk.DoubleVar(value=33.0)
        self.subtract_ref_var = tk.BooleanVar(value=False)
        self.show_vgen_var    = tk.BooleanVar(value=True)
        self.ref_calc_var     = tk.BooleanVar(value=False)
        self._cs_optimized = False
        self._ga_running   = False
        self._fast_eval    = False

        def L(parent, text, **kw):
            return ttk.Label(parent, text=text, font=SF, **kw)
        def E(parent, var, w=9):
            return ttk.Entry(parent, textvariable=var, width=w)

        PX = (8, 2)  # left padx for right sub-columns

        # Row 0: V_gen | Rgen
        L(params_frame, "V_gen (mV RMS):").grid(row=0, column=0, sticky="w", padx=(0,2), pady=1)
        E(params_frame, self.V_gen_rms).grid(row=0, column=1, padx=2, pady=1)
        L(params_frame, "Rgen / Z0 port1 (Ω):").grid(row=0, column=2, sticky="w", padx=PX, pady=1)
        E(params_frame, self.R_internal_gen).grid(row=0, column=3, padx=2, pady=1)

        # Row 1: matched load checkbox | Port 2 Z0
        ttk.Checkbutton(params_frame, text="Gen assumes 50Ω load",
                        variable=self.gen_matched_load).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=1)
        L(params_frame, "Port 2 Z0 / Meas (Ω):").grid(row=1, column=2, sticky="w", padx=PX, pady=1)
        E(params_frame, self.Z_in_measure).grid(row=1, column=3, padx=2, pady=1)

        # Row 2: Rdiv1 | Rdiv2
        L(params_frame, "Rdiv1 Series (Ω):").grid(row=2, column=0, sticky="w", padx=(0,2), pady=1)
        E(params_frame, self.R_div1).grid(row=2, column=1, padx=2, pady=1)
        L(params_frame, "Rdiv2 Shunt (Ω):").grid(row=2, column=2, sticky="w", padx=PX, pady=1)
        E(params_frame, self.R_div2).grid(row=2, column=3, padx=2, pady=1)

        # Row 3: Rser | Lload
        L(params_frame, "Rser Branch (Ω):").grid(row=3, column=0, sticky="w", padx=(0,2), pady=1)
        E(params_frame, self.Rser).grid(row=3, column=1, padx=2, pady=1)
        L(params_frame, "Lload (H):").grid(row=3, column=2, sticky="w", padx=PX, pady=1)
        E(params_frame, self.Lload).grid(row=3, column=3, padx=2, pady=1)

        # Row 4: Rload | Cload
        L(params_frame, "Rload (Ω):").grid(row=4, column=0, sticky="w", padx=(0,2), pady=1)
        E(params_frame, self.Rload).grid(row=4, column=1, padx=2, pady=1)
        L(params_frame, "Cload (F):").grid(row=4, column=2, sticky="w", padx=PX, pady=1)
        E(params_frame, self.Cload).grid(row=4, column=3, padx=2, pady=1)

        # Row 5: Reference calculation mode
        ttk.Checkbutton(params_frame, text="Reference Calculation (resonators OFF)",
                        variable=self.ref_calc_var).grid(
            row=5, column=0, columnspan=4, sticky="w", padx=(0,2), pady=(4,1))

        # ── Resonator Grid ─────────────────────────────────────────────────
        res_params_frame = ttk.LabelFrame(main_frame,
                                          text="Editable Resonator Parameters",
                                          padding="4")
        res_params_frame.grid(row=2, column=0, sticky="nsew", pady=(0,4))

        self.res_params_grid = ttk.Frame(res_params_frame)
        self.res_params_grid.pack(fill="both", expand=True)
        self.resonator_vars = []
        self.resonator_entry_widgets = []

        # ── Fundamental frequency ──────────────────────────────────────────
        fund_frame = ttk.Frame(main_frame, padding="4")
        fund_frame.grid(row=3, column=0, sticky="ew", pady=(2, 2))
        ttk.Label(fund_frame, text="Fundamental (MHz):",
                  font=("TkDefaultFont", 16, "bold")).pack(side=tk.LEFT, padx=(0, 8))
        tk.Entry(fund_frame, textvariable=self.f_fundamental,
                 width=8, font=("TkDefaultFont", 16)).pack(side=tk.LEFT)

        # ── Status bar ─────────────────────────────────────────────────────
        status_frame = ttk.Frame(main_frame, padding="2")
        status_frame.grid(row=4, column=0, sticky="ew", pady=(0,2))
        status_frame.columnconfigure(1, weight=1)

        self._led_canvas = tk.Canvas(status_frame, width=12, height=12,
                                     highlightthickness=0)
        self._led_canvas.grid(row=0, column=0, padx=(0,4))
        self._led = self._led_canvas.create_oval(1, 1, 11, 11, fill="#cccccc", outline="")
        self._led_on  = False
        self._led_job = None
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(status_frame, textvariable=self.status_var,
                  font=SF).grid(row=0, column=1, sticky="ew")

        # Stop button lives in status bar (for GA)
        self._stop_btn = ttk.Button(status_frame, text="■ Stop",
                                    command=self._stop_ga, state="disabled")
        self._stop_btn.grid(row=0, column=2, padx=(8,0))

        # ── Version footer ─────────────────────────────────────────────────
        ttk.Label(main_frame,
                  text=f"Resonator Optimizer Ver. {VERSION}",
                  font=("Helvetica", 12, "bold"), foreground="blue"
                  ).grid(row=5, column=0, pady=(2,4))

        # Traces — after all vars are defined
        self.Cload.trace_add("write", self._invalidate_cs)
        self.Rload.trace_add("write", self._invalidate_cs)
        self.f_fundamental.trace_add("write", self._invalidate_cs)

        # Plot windows are created on demand as separate Toplevel windows
        self._plot_wins   = {}   # key -> Toplevel
        self._plot_figs   = {}   # key -> Figure
        self._plot_canvas = {}   # key -> FigureCanvasTkAgg
        # Legacy references so GA optimiser code (self.fig/self.canvas) still works
        self.fig    = None
        self.canvas = None
        self.axes   = None

        self.toggle_analysis_inputs()
        self._load_resonators()
        self.last_results_df   = None
        self.last_filename     = None
        self._ga_last_gen_best = []
        self._ga_last_gen_mean = []
        self._ga_last_names    = []
        self._ga_last_fom      = 0.0
        # Compat shims so GA worker code still compiles
        self._opt_menu              = type("FakeMenu", (), {
            "entryconfigure": lambda *a, **k: None})()
        self._opt_menu_stop_index   = 0
        # Shim — some paths reference _run_btn; status bar has no run btn now
        self._run_btn = type("B", (), {"configure": lambda *a,**k: None})()
        # analysis_choice: plain StringVar, set before each run
        self.analysis_choice = tk.StringVar(value="Steady State")

    def _quit_app(self):
        for win in self._plot_wins.values():
            try: win.destroy()
            except Exception: pass
        self.root.quit()
        self.root.destroy()

    # -----------------------------------------------------------------------
    # Menu: Files
    # -----------------------------------------------------------------------
    def _menu_open_stimulus(self):
        filepath = filedialog.askopenfilename(
            initialdir=self.script_dir, title="Open Stimulus File",
            filetypes=(("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")))
        if filepath:
            self.stimulus_file.set(filepath)
            self.status_var.set(f"Stimulus: {os.path.basename(filepath)}")

    def _menu_open_resonators(self):
        filepath = filedialog.askopenfilename(
            initialdir=self.script_dir, title="Open Resonators File",
            filetypes=(("RES files", "*.res"), ("Text files", "*.txt"), ("All files", "*.*")))
        if filepath:
            self.resonators_file.set(filepath)
            self._load_resonators()

    def _menu_open_inductor_db(self):
        filepath = filedialog.askopenfilename(
            initialdir=self.script_dir, title="Open Inductor Database (CSV)",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
        if not filepath:
            return
        try:
            df = pd.read_csv(filepath)
            required = {"Part_Number", "R1_Ohm", "R2_Ohm", "C_pF", "L_nH", "k", "Upper_limit_MHz"}
            if not required.issubset(set(df.columns)):
                messagebox.showerror("Invalid File",
                    f"CSV must contain columns:\n{', '.join(sorted(required))}")
                return
            self.inductor_db = df
            self.inductor_db_path.set(os.path.basename(filepath))
            self.status_var.set(f"Loaded {len(df)} inductors from {os.path.basename(filepath)}")
            self._rebuild_inductor_menu()
            # Default Pop = DB size; update combobox if already open
            K = len(df)
            self.ga_pop.set(K)
            if self._pop_combobox is not None:
                vals = sorted(set([K, 10, 20, 30, 50, 100]) |
                              ({K} if K > 0 else set()))
                self._pop_combobox.configure(values=vals)
                self._pop_combobox.set(K)
        except Exception as e:
            messagebox.showerror("Error loading database", str(e))

    def _rebuild_inductor_menu(self):
        self.inductor_menu.delete(0, "end")
        if self.inductor_db is None or self.inductor_db.empty:
            self.inductor_menu.add_command(label="(No database loaded)", state="disabled")
            return
        self.inductor_menu.add_command(
            label="Select Inductors…", command=self._open_inductor_selector)

    # -----------------------------------------------------------------------
    # Inductor Selector Dialog  — checkbox-based, order-aware
    # -----------------------------------------------------------------------
    def _open_inductor_selector(self):
        if self.inductor_db is None:
            messagebox.showinfo("No Database", "Please load an inductor database first.")
            return

        dlg = tk.Toplevel(self.root)
        dlg.title("Choose Inductors — click checkbox to select (order matters)")
        dlg.grab_set()
        dlg.resizable(True, True)
        dlg.geometry("820x540")

        cols = list(self.inductor_db.columns)
        rows = self.inductor_db.to_dict("records")

        selection_order = []   # list of row indices in click order
        summary_var = tk.StringVar(value="0 inductors selected")

        COL_BG_EVEN  = "#f5f5f5"
        COL_BG_ODD   = "#ffffff"
        COL_BG_SEL   = "#d0e8ff"   # highlight for selected rows
        BADGE_BG     = "#2a7ae2"
        BADGE_EMPTY  = "#cccccc"

        # Per-row widgets stored so we can update them
        num_labels  = []   # tk.Label showing order number
        row_widgets = []   # list of all tk widgets in each data row (for bg recolor)

        def refresh(changed_idx=None):
            """Update badges, row backgrounds, summary."""
            for i in range(len(rows)):
                pos = selection_order.index(i) + 1 if i in selection_order else 0
                if pos:
                    num_labels[i].config(text=str(pos),
                                         bg=BADGE_BG, fg="white",
                                         font=("Helvetica", 9, "bold"))
                else:
                    num_labels[i].config(text="", bg=BADGE_EMPTY, fg=BADGE_EMPTY)

                # Row background
                base = COL_BG_EVEN if i % 2 == 0 else COL_BG_ODD
                bg = COL_BG_SEL if i in selection_order else base
                for w in row_widgets[i]:
                    try:
                        w.config(bg=bg)
                    except tk.TclError:
                        pass

            n = len(selection_order)
            names = [rows[i]["Part_Number"] for i in selection_order]
            summary_var.set(f"{n} selected: {', '.join(names)}" if n
                            else "0 inductors selected")

        def toggle(i):
            if i in selection_order:
                selection_order.remove(i)
            else:
                selection_order.append(i)
            refresh(i)

        # ── scrollable table ───────────────────────────────────────────────
        outer  = ttk.Frame(dlg)
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

        # ── header ──
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
            tk.Label(inner, text=c, width=col_widths.get(c, 8), font=HDR,
                     relief="ridge", anchor="center",
                     bg="#dde8f5").grid(row=0, column=ci+2, padx=1, pady=1, sticky="nsew")

        # ── data rows ──
        # Each row has a tk.BooleanVar so checkbutton state is reliable.
        # toggle() is the single place that mutates selection_order.
        cb_vars = [tk.BooleanVar(value=False) for _ in rows]

        def toggle(i):
            """Toggle row i in/out of selection_order, then refresh display."""
            if i in selection_order:
                selection_order.remove(i)
                cb_vars[i].set(False)
            else:
                selection_order.append(i)
                cb_vars[i].set(True)
            refresh()

        for ri, row in enumerate(rows):
            base_bg = COL_BG_EVEN if ri % 2 == 0 else COL_BG_ODD
            wlist = []

            # Column 0: order-number badge
            nl = tk.Label(inner, text="", width=3, anchor="center",
                          bg=BADGE_EMPTY, fg=BADGE_EMPTY,
                          font=("Helvetica", 9, "bold"), relief="flat")
            nl.grid(row=ri+1, column=0, padx=1, pady=1, sticky="nsew")
            num_labels.append(nl)
            wlist.append(nl)

            # Column 1: checkbutton — variable drives visual; command drives order
            cb = tk.Checkbutton(inner, variable=cb_vars[ri],
                                bg=base_bg, activebackground=base_bg,
                                command=lambda i=ri: toggle(i))
            # Because command fires AFTER tk flips the var, we must reconcile:
            # wrap toggle so it does NOT re-flip the var (it was already flipped)
            def _cmd(i=ri):
                # var was already toggled by tk; sync selection_order to match
                if cb_vars[i].get():
                    if i not in selection_order:
                        selection_order.append(i)
                else:
                    if i in selection_order:
                        selection_order.remove(i)
                refresh()
            cb.config(command=_cmd)
            cb.grid(row=ri+1, column=1, padx=4, pady=1, sticky="nsew")
            wlist.append(cb)

            for ci, c in enumerate(cols):
                val = row[c]
                txt = f"{val:.4g}" if isinstance(val, float) else str(val)
                lbl = tk.Label(inner, text=txt, width=col_widths.get(c, 8),
                               anchor="center", bg=base_bg,
                               font=("Helvetica", 9))
                lbl.grid(row=ri+1, column=ci+2, padx=1, pady=1, sticky="nsew")
                # Clicking anywhere on the row also toggles
                def _row_click(e, i=ri):
                    cb_vars[i].set(not cb_vars[i].get())
                    if cb_vars[i].get():
                        if i not in selection_order:
                            selection_order.append(i)
                    else:
                        if i in selection_order:
                            selection_order.remove(i)
                    refresh()
                lbl.bind("<Button-1>", _row_click)
                wlist.append(lbl)

            row_widgets.append(wlist)

        def clear_selection():
            selection_order.clear()
            for v in cb_vars:
                v.set(False)
            refresh()

        # ── summary + buttons ──────────────────────────────────────────────
        ttk.Label(dlg, textvariable=summary_var,
                  foreground="darkblue",
                  font=("Helvetica", 9, "bold"),
                  padding=4).pack(anchor="w")

        btn_frame = ttk.Frame(dlg, padding=6)
        btn_frame.pack(fill="x")

        def confirm():
            if not selection_order:
                messagebox.showwarning("Nothing selected",
                                       "Please check at least one inductor.",
                                       parent=dlg)
                return
            self.selected_inductors = [rows[i] for i in selection_order]
            self._apply_inductors_to_resonator_grid()
            dlg.destroy()

        def clear_all():
            clear_selection()

        ttk.Button(btn_frame, text="Add to Resonator Grid",
                   command=confirm).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Clear All",
                   command=clear_all).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Cancel",
                   command=dlg.destroy).pack(side="left", padx=5)

        # Unbind mousewheel when dialog closes
        dlg.protocol("WM_DELETE_WINDOW",
                     lambda: (canvas.unbind_all("<MouseWheel>"), dlg.destroy()))

    def _apply_inductors_to_resonator_grid(self):
        """
        Map database columns to resonator grid:
          R1_Ohm -> R1,  R2_Ohm -> R_DC,  C_pF -> C1 (F),  L_nH -> L (H),  k -> k
        Cs/Cp handling:
          - If a previous grid exists, prompt "Keep Cs/Cp values?"
            - YES: carry over existing Cs/Cp by row index; new rows beyond old count
                   get defaults (Cs=1e-12 F, Cp=0.5e-12 F).
            - NO:  all rows get defaults.
        """
        if not self.selected_inductors:
            return

        NEW_DEFAULT_Cs = 1.0e-12
        NEW_DEFAULT_Cp = 0.5e-12
        n_new = len(self.selected_inductors)

        # Collect existing Cs/Cp from the current grid (if any)
        existing_cs_cp = []   # list of (Cs_F, Cp_F) per current row
        if hasattr(self, 'resonator_entry_widgets') and self.resonator_entry_widgets:
            for row_w in self.resonator_entry_widgets:
                try:
                    cs = float(row_w[5].get()) * 1e-12   # col 5 = Cs in pF
                    cp = float(row_w[6].get()) * 1e-12   # col 6 = Cp in pF
                except (ValueError, IndexError):
                    cs, cp = NEW_DEFAULT_Cs, NEW_DEFAULT_Cp
                existing_cs_cp.append((cs, cp))

        # Decide whether to keep existing Cs/Cp
        keep_cs = False
        if existing_cs_cp:
            n_old = len(existing_cs_cp)
            if n_new != n_old:
                msg = (
                    f"Number of resonators changed ({n_old} -> {n_new}).\n\n"
                    f"Keep existing Cs/Cp for the first {min(n_old, n_new)} rows?\n"
                    f"New rows (if any) will use defaults: "
                    f"Cs={NEW_DEFAULT_Cs*1e12:.4g} pF, Cp={NEW_DEFAULT_Cp*1e12:.4g} pF.")
            else:
                msg = (
                    f"Keep existing Cs/Cp values for all {n_new} rows?\n\n"
                       f"(No = reset to defaults: "
                       f"Cs={NEW_DEFAULT_Cs*1e12:.4g} pF, Cp={NEW_DEFAULT_Cp*1e12:.4g} pF)")
            keep_cs = messagebox.askyesno("Cs / Cp Assignment", msg)

        # Build new parameter rows
        new_rows = []
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
            new_rows.append([R1, C1, L, k, R_DC, Cs, Cp])

        params = np.array(new_rows)
        self.resonator_params_loaded = params.copy()
        self._populate_resonator_grid(params)
        names = [d["Part_Number"] for d in self.selected_inductors]
        self.status_var.set(f"Loaded {len(names)} inductor(s): {', '.join(names)}")

    def toggle_analysis_inputs(self):
        pass  # All inputs now inline; no separate frames to enable/disable
    
    def to_time_domain(self, phasor, omega, t):
        return np.abs(phasor) * np.cos(omega * t + np.angle(phasor))

    def _load_stimulus(self):
        path_str = self.stimulus_file.get()
        if os.path.exists(path_str):
            filename = path_str
        else:
            filename = os.path.join(self.script_dir, path_str)
            
        try:
            with open(filename, 'r') as f:
                mags = [float(val) for val in f.readline().strip().split(',')]
                phases = [float(val) for val in f.readline().strip().split(',')]
            return mags, phases, len(mags) - 1
        except Exception as e:
            print(f"Error loading stimulus {filename}: {e}. Using defaults.")
            return [0.518, 0.634, 0.0365, 0.2048, 0.035], [180, 180, 0, 0], 4

    def _load_resonators(self):
        path_str = self.resonators_file.get()
        if os.path.exists(path_str):
            filename = path_str
        else:
            filename = os.path.join(self.script_dir, path_str)

        try:
            params = np.loadtxt(filename, delimiter=",")
            if params.ndim == 1: params = params.reshape(1, -1)
            self.status_var.set(f"Loaded resonators from {os.path.basename(filename)}")
        except Exception as e:
            print(f"Error loading resonators {filename}: {e}. Using defaults.")
            self.status_var.set(f"Error loading file. Used defaults.")
            params = np.array([[60.0,0.604e-12,2.7e-6,2.8e-3,3.0,3.66e-12,0.5e-12],[60.0,0.288e-12,1.2e-6,9.7e-4,3.0,2.0e-12,0.5e-12],[66.0,0.258e-12,0.56e-6,5.54e-4,1.9,1.9e-12,0.5e-12],[66.0,0.096e-12,0.32e-6,4.74e-4,1.4,2.1e-12,0.5e-12]])
            
        self.resonator_params_loaded = params.copy()
        self._populate_resonator_grid(self.resonator_params_loaded)
        return self.resonator_params_loaded

    def _populate_resonator_grid(self, params, filter_by_ratio=False):
        for widget in self.res_params_grid.winfo_children(): widget.destroy()
        # Always drop inactive sentinels (Cs < 1e-16 = optimizer couldn't find valid Cs)
        # Only apply min-ratio filter when displaying GA/Greedy search results
        if len(params) > 0:
            if filter_by_ratio:
                # Leave-one-out: keep resonator j only if removing it drops FOM
                # by at least ga_min_contrib percentage points.
                MIN_CONTRIB = self.ga_min_contrib.get()
                try:
                    mags_d, phases_d, _ = self._load_stimulus()
                except Exception:
                    mags_d, phases_d = [1.0], [0.0]
                f0_d     = self._get_f_fundamental() * 1e6
                Rload_d  = self.Rload.get();  Cload_d = self.Cload.get()
                Lload_d  = self.Lload.get();  Rser_d  = self.Rser.get()
                R_div2_d = self.R_div2.get(); R_div1_d= self.R_div1.get()
                R_gen_d  = self.R_internal_gen.get()
                Z_in_d   = self.Z_in_measure.get()
                full_p   = np.array([r for r in params if float(r[5]) >= 1e-16])
                if len(full_p) == 0:
                    params = np.zeros((0, 7))
                else:
                    fom_full = self._compute_fom(
                        full_p, f0_d, mags_d, phases_d,
                        R_gen_d, R_div1_d, R_div2_d, 0.0, Z_in_d,
                        Rser_d, Rload_d, Cload_d, Lload_d)
                    keep = []
                    for j in range(len(full_p)):
                        others = np.delete(full_p, j, axis=0)
                        fom_without = self._compute_fom(
                            others, f0_d, mags_d, phases_d,
                            R_gen_d, R_div1_d, R_div2_d, 0.0, Z_in_d,
                            Rser_d, Rload_d, Cload_d, Lload_d) if len(others) else 0.0
                        if (fom_full - fom_without) >= MIN_CONTRIB:
                            keep.append(full_p[j])
                    params = np.array(keep) if keep else np.zeros((0, full_p.shape[1]))
            else:
                # Just drop inactive sentinels
                params = np.array([r for r in params if float(r[5]) >= 1e-16])                          if len(params) > 0 else params
        if len(params) == 0:
            params = np.zeros((0, 7))
        # Store Entry widgets directly — no DoubleVar aliasing issues
        self.resonator_entry_widgets = []   # list-of-lists of ttk.Entry
        headers = ("#", "R1", "C1 (F)", "L (H)", "k", "R_DC", "Cs (pF)", "Cp (pF)")
        for col, text in enumerate(headers):
            ttk.Label(self.res_params_grid, text=text,
                      font='TkDefaultFont 9 bold').grid(row=0, column=col, padx=2, pady=2)
        for idx, row_data in enumerate(params):
            ttk.Label(self.res_params_grid,
                      text=f"{idx+1}").grid(row=idx+1, column=0)
            # Columns in display order: R1, C1(F), L(H), k, R_DC, Cs(pF), Cp(pF)
            display_vals = [
                float(row_data[0]),          # R1      — Ohm, as-is
                float(row_data[1]),          # C1      — Farads, as-is
                float(row_data[2]),          # L       — Henrys, as-is
                float(row_data[3]),          # k       — as-is
                float(row_data[4]),          # R_DC    — Ohm, as-is
                float(row_data[5]) * 1e12,   # Cs      — stored F, displayed pF
                float(row_data[6]) * 1e12,   # Cp      — stored F, displayed pF
            ]
            row_widgets = []
            for col_i, dval in enumerate(display_vals):
                e = ttk.Entry(self.res_params_grid, width=8)
                e.insert(0, f"{dval:.6g}")
                e.grid(row=idx+1, column=col_i+1)
                row_widgets.append(e)
            self.resonator_entry_widgets.append(row_widgets)
        # Legacy alias so any other code using resonator_vars still works
        self.resonator_vars = self.resonator_entry_widgets

    def _get_resonator_params_from_gui(self):
        """Read resonator params from Entry widget text in displayed row order."""
        ref_mode = self.ref_calc_var.get()
        rows = []
        for row_widgets in self.resonator_entry_widgets:
            # cols: R1, C1(F), L(H), k, R_DC, Cs(pF), Cp(pF)
            raw = [float(w.get()) for w in row_widgets]
            raw[5] *= 1e-12   # Cs: pF -> F
            raw[6] *= 1e-12   # Cp: pF -> F
            if ref_mode:
                raw[5] = 1e-21  # 1e-9 pF — below inactive threshold, resonators open-circuit
            rows.append(raw)
        return np.array(rows)

    def _save_resonators(self):
        try:
            params = self._get_resonator_params_from_gui()
            initial_dir = self.default_save_dir if os.path.exists(self.default_save_dir) else self.script_dir
            filepath = filedialog.asksaveasfilename(defaultextension=".res", filetypes=[("RES files", "*.res")], initialdir=initial_dir)
            if filepath:
                np.savetxt(filepath, params, delimiter=",", fmt='%.6e')
                self.status_var.set(f"Saved to {os.path.basename(filepath)}")
                self.resonators_file.set(filepath)
        except Exception as e: messagebox.showerror("Error", str(e))

    def _build_save_filename(self, analysis_type, f0_hz, res_params, fom=None):
        """
        Build a structured filename for analysis output.

        analysis_type : 'freq' or 'steady'
        f0_hz         : fundamental frequency in Hz
        res_params    : numpy array of resonator params (may be empty)
        fom           : FOM in % (steady-state only)

        Frequency:   Spec_49MHz-820-680-330_v1.dat   /  Spec_49MHz-ref.dat
        Steady:      SS-BP3_16-FOM75.3-49MHz-820-680-330_v1.dat
                     SS-BP3_16-FOMref-49MHz-ref.dat
        """
        import re

        f0_mhz = round(f0_hz / 1e6)
        n_res  = len(res_params) if res_params is not None and len(res_params) else 0

        # ── L values in nH (shared by both modes) ────────────────────────
        if n_res > 0 and not self.ref_calc_var.get():
            ls_str = '-'.join(str(round(p[2] * 1e9)) for p in res_params)
        else:
            ls_str = None   # reference mode → replaced by 'ref' suffix

        # ── Frequency analysis: Spec_(f0)MHz-... ─────────────────────────
        if analysis_type == 'freq':
            if ls_str is None:
                base = f"Spec_{f0_mhz}MHz-ref"
                return f"{base}.dat"
            base = f"Spec_{f0_mhz}MHz-{ls_str}"

        # ── Steady state: SS-{stim_abbrev}{stim_nums}-(f0)MHz-... ────────
        else:
            stim_base = os.path.splitext(
                os.path.basename(self.stimulus_file.get()))[0]
            tokens = re.split(r'[\s_]+', stim_base)
            # Text abbreviation: first letter of each non-numeric, non-"=" token
            abbrev = ''.join(
                t[0].upper()
                for t in tokens
                if t and '=' not in t and not t.lstrip('-').isdigit()
            )[:4] or 'AN'
            # Numeric values: digits-only tokens, excluding any with "="
            nums = '_'.join(
                t for t in tokens
                if t.lstrip('-').isdigit() and '=' not in t
            )
            stim_tag = f"{abbrev}{nums}" if nums else abbrev

            fom_tag = f"FOM{fom:.1f}" if fom is not None else "FOM?"
            if ls_str is None:
                base = f"SS-{stim_tag}-{fom_tag}-{f0_mhz}MHz-ref"
                return f"{base}.dat"
            base = f"SS-{stim_tag}-{fom_tag}-{f0_mhz}MHz-{ls_str}"

        # ── Auto-increment version ────────────────────────────────────────
        save_dir = (self.default_save_dir
                    if os.path.isdir(self.default_save_dir) else '.')
        ver = 1
        while os.path.exists(os.path.join(save_dir, f"{base}_v{ver}.dat")):
            ver += 1
        return f"{base}_v{ver}.dat"

    def _save_data(self):
        if self.last_results_df is not None:
            try:
                default_name = self.last_filename if self.last_filename else "data.dat"

                filepath = filedialog.asksaveasfilename(
                    defaultextension=".dat",
                    initialfile=default_name,
                    filetypes=[("DAT files", "*.dat"), ("CSV files", "*.csv")],
                    initialdir=self.default_save_dir
                )

                if filepath:
                    self.last_results_df.to_csv(filepath, index=False)
                    self.status_var.set(f"Saved to {os.path.basename(filepath)}")
            except Exception as e: messagebox.showerror("Error", str(e))

    # -----------------------------------------------------------------------
    # Cs OPTIMIZATION  — maximise current ratio |Iload/Iser| at each harmonic
    # -----------------------------------------------------------------------
    def _get_f_fundamental(self):
        """Return fundamental frequency in MHz, falling back to 33.0 on bad input."""
        try:
            return float(self.f_fundamental.get())
        except (tk.TclError, ValueError):
            self.f_fundamental.set(33.0)
            return 33.0

    def _invalidate_cs(self, *_):
        """Mark Cs stale when Cload, Rload, or Fund. changes after optimization."""
        if self._cs_optimized:
            self._cs_optimized = False
            self.status_var.set(
                "⚠  Cload / Rload / Fund. changed — Cs values are stale. Re-run Optimize Cs.")
            try:
                self._opt_btn.configure(style="Warn.TButton")
            except Exception:
                pass

    def _current_ratio_at(self, f, params_active, Rload, Cload, Lload, Rser, R_div2):
        """
        Compute |Iload/Iser| at frequency f.
        |Iload/Iser| = |Z_struct / Z_load| where Z_struct = Z_res_total || Z_load.
        >1 means resonators are diverting current into the load.
        """
        w = 2.0 * np.pi * f
        if w == 0:
            return 0.0
        Y_res = sum(1.0 / self.calculate_single_resonator_impedance(f, p)
                    for p in params_active if p[5] >= 1e-16)
        Z_res_total = 1.0 / Y_res if abs(Y_res) > 1e-30 else np.inf
        Z_load = Rload + 1j * w * Lload + 1.0 / (1j * w * Cload)
        if Z_res_total == np.inf:
            Z_struct = Z_load
        else:
            Z_struct = 1.0 / (1.0 / Z_load + 1.0 / Z_res_total)
        return abs(Z_struct / Z_load) if abs(Z_load) > 1e-30 else 0.0

    def _cs_for_max_ratio(self, params_active, row_idx, f_target,
                          Rload, Cload, Lload, Rser, R_div2):
        """
        Scan Cs for resonator row_idx to maximise |Iload/Iser| at f_target.
        Uses a two-pass grid search (coarse then fine) — no scipy needed.
        """
        p_work = params_active.copy()
        def neg_ratio(log_cs):
            p_work[row_idx, 5] = 10.0 ** log_cs
            return -self._current_ratio_at(f_target, p_work,
                                           Rload, Cload, Lload, Rser, R_div2)

        # The resonator presents high impedance (parallel anti-resonance) when the
        # series block Z_AB+1/jωCs is net inductive, i.e. when Cs > Cs_series_res
        # = 1/(L*(2π*f_target)²).  Search from that point upward.
        cs_min = 0.001e-12   # FIX-2: lower floor (was 0.01 pF)
        cs_max = 1000e-12    # hard cap: no Cs > 1000 pF
        L = params_active[row_idx, 2]
        if L > 0:
            cs_series_res = 1.0 / (L * (2.0 * np.pi * f_target) ** 2)
            # FIX-2: start 1 decade below simplified estimate; actual series
            # resonance of Z_series_block (with C1 in Z_AB_parallel) can differ
            cs_min = max(0.001e-12, cs_series_res * 0.1)
        cs_max = min(cs_max, 1000e-12)   # enforce hard cap regardless of cs_min

        n_coarse = 60  if self._fast_eval else 500
        n_fine   = 30  if self._fast_eval else 200
        log_cs_grid = np.linspace(np.log10(cs_min), np.log10(cs_max), n_coarse)
        neg_vals = np.array([neg_ratio(lc) for lc in log_cs_grid])
        best_idx = int(np.argmin(neg_vals))
        lo = log_cs_grid[max(0, best_idx - 2)]
        hi = log_cs_grid[min(len(log_cs_grid) - 1, best_idx + 2)]
        fine_grid = np.linspace(lo, hi, n_fine)
        fine_vals = np.array([neg_ratio(lc) for lc in fine_grid])
        best_log_cs = fine_grid[int(np.argmin(fine_vals))]
        best_cs = 10.0 ** best_log_cs
        if best_cs < 1e-16:
            return 1e-17   # FIX-2: inactive threshold matches model (was 0.01 pF)
        # If even the best Cs gives ratio ≤ 1 this component doesn't help at
        # f_target — mark inactive so it gets pruned rather than parking at
        # 1000 pF and pulling a subharmonic into the solution.
        best_ratio = -neg_ratio(best_log_cs)
        if best_ratio <= 1.0:
            return 1e-17   # useless at this harmonic → inactive
        # FIX-6: sub-harmonic reject — discard if anti-resonance falls below f0.
        # f_ar ≈ sqrt((Cs+Cp)/(L·Cs·Cp)) / 2π  (simplified, ignores R1/C1).
        # If f_ar < f0 the resonator blocks below the fundamental — not useful.
        L_val  = params_active[row_idx, 2]
        Cp_val = params_active[row_idx, 6]
        if L_val > 0 and Cp_val > 0 and best_cs > 0:
            f_ar   = np.sqrt((best_cs + Cp_val) / (L_val * best_cs * Cp_val)) / (2 * np.pi)
            f0_est = f_target / (row_idx + 1)   # harmonic index → fundamental
            if f_ar < f0_est:
                return 1e-17   # sub-harmonic resonator → reject
        return best_cs

    def _joint_optimise_cs(self, params, n_active, f0,
                           Rload, Cload, Lload, Rser, R_div2, reseed=True):
        """
        Jointly re-optimise Cs[0..n_active-1] using coordinate descent.

        reseed=True  (default): Step 1 reinitialises all Cs from series-resonance
                     values before optimising.  Use this for a full optimisation.
        reseed=False: skip Step 1 and run coordinate descent from the current Cs
                     values.  Use this for a gentle tweak (e.g. after Rser scan)
                     so existing tuning is preserved and only small adjustments
                     are made.

        Step 1 — initialise (reseed=True only): set Cs[i] to 2× series-resonance
                 for harmonic (i+1)*f0, then run a forward sequential pass.

        Step 2 — sequential pass: optimise Cs[0] at f0, then Cs[1] at 2*f0, etc.

        Step 3 — coordinate-descent rounds: cycle until convergence.
        """
        if n_active == 0:
            return

        if reseed:
            # ── Step 1: seed Cs just above series resonance for each harmonic ─
            # We want parallel anti-resonance (high Z), which requires Cs > cs_series_res
            # so the L+Cs branch is net inductive at f_target.  Seed at 2x cs_series_res.
            for i in range(n_active):
                L = params[i, 2]
                f_target = (i + 1) * f0
                if L > 0:
                    cs_series_res = 1.0 / (L * (2.0 * np.pi * f_target) ** 2)
                    params[i, 5] = np.clip(cs_series_res * 2.0, 0.001e-12, 1000e-12)  # FIX-2: lower floor
                else:
                    params[i, 5] = 1e-12   # fallback

        # ── Step 2: sequential forward pass — each resonator sees correctly
        #            initialised neighbours ────────────────────────────────────
        # Pass only params[:i+1] (not all n_active) so that higher harmonics
        # whose Cs is not yet optimised do not perturb the search for resonator i.
        # This guarantees resonator 0 searches at f0, resonator 1 at 2*f0, etc.
        for i in range(n_active):
            params[i, 5] = self._cs_for_max_ratio(
                params[:i + 1], i, (i + 1) * f0,
                Rload, Cload, Lload, Rser, R_div2)

        # ── Step 3: coordinate-descent rounds until convergence ───────────────
        MAX_ROUNDS = 2 if self._fast_eval else 4
        for _ in range(MAX_ROUNDS):
            prev = params[:n_active, 5].copy()
            for i in range(n_active):
                params[i, 5] = self._cs_for_max_ratio(
                    params[:n_active], i, (i + 1) * f0,
                    Rload, Cload, Lload, Rser, R_div2)
            if np.all(np.abs(params[:n_active, 5] - prev) /
                      np.maximum(prev, 1e-20) < 0.005):
                break

    def _update_grid_cs(self, params):
        """Write Cs values from params array back into the Entry widgets."""
        for i, row_w in enumerate(self.resonator_entry_widgets):
            if i >= len(params):
                break
            row_w[5].delete(0, tk.END)
            row_w[5].insert(0, f"{params[i, 5] * 1e12:.6g}")

    def _run_cs_optimization(self):
        if self.ref_calc_var.get():
            messagebox.showinfo("Reference Mode",
                                "Disable Reference Calculation mode before optimising Cs.")
            return
        if not hasattr(self, '_opt_btn') or self._opt_btn is None:
            self._opt_btn = type('_FakeBtn', (), {'configure': lambda self, **k: None})()  # noqa
        """
        Run Cs optimisation in a background thread so each step is plotted
        immediately as it completes. Uses a queue to pass plot data back to
        the main (tkinter) thread safely.
        """
        params = self._get_resonator_params_from_gui()
        N = len(params)
        if N == 0:
            messagebox.showwarning("No resonators", "Load resonators first.")
            return

        # Disable button while running
        self._led_start()
        self._opt_btn.configure(state="disabled", style="TButton")

        f0     = self._get_f_fundamental() * 1e6
        Rload  = self.Rload.get()
        Cload  = self.Cload.get()
        Lload  = self.Lload.get()
        Rser   = self.Rser.get()
        R_div2 = self.R_div2.get()


        colors  = plt.cm.tab10(np.linspace(0, 1, max(N, 1)))
        f_MHz   = np.linspace(max(f0 * 0.3, 1e6), (N + 1.5) * f0, 2000) / 1e6
        f_plot  = f_MHz * 1e6

        # ── Build the axes and ALL line objects up front ──────────────────
        # We create one line per step for ratio, one per resonator for Z,
        # then update them with set_data() — no cla(), no tight_layout() in loop.
        fig, axes, canvas = self._get_plot_window(
            'cs_opt', 'Cs Optimisation', (8, 6), 2)
        self.fig, self.canvas = fig, canvas
        ax_ratio, ax_Z = axes[0], axes[1]

        ax_ratio.set_xlabel("Frequency (MHz)")
        ax_ratio.set_ylabel("|Iload / Iser|")
        ax_ratio.set_title("Current Ratio — Cs Optimisation Progress")
        ax_ratio.grid(True, alpha=0.4)

        ax_Z.set_xlabel("Frequency (MHz)")
        ax_Z.set_ylabel("|Z_resonator| (Ω)")
        ax_Z.set_title("Individual Resonator Impedance")
        ax_Z.grid(True, alpha=0.4)
        ax_Z.set_yscale('log')

        # Pre-create N ghost ratio lines (grey, thin) + N bold ratio lines
        ghost_lines = []
        bold_lines  = []
        for k in range(N):
            gl, = ax_ratio.plot([], [], color='#cccccc', linewidth=0.8,
                                zorder=1, visible=False)
            bl, = ax_ratio.plot([], [], color=colors[k], linewidth=2.0,
                                zorder=3, visible=False,
                                label=f"Step {k+1}: +L{k+1}")
            ghost_lines.append(gl)
            bold_lines.append(bl)

        # Harmonic vlines on ratio axis (one per resonator)
        harm_vlines_ratio = []
        for i in range(N):
            vl = ax_ratio.axvline((i+1)*f0/1e6, color=colors[i],
                                  linestyle='--', linewidth=1.0,
                                  alpha=0.0)   # invisible until that step
            harm_vlines_ratio.append(vl)

        # Pre-create N resonator Z lines
        Z_lines = []
        harm_vlines_Z = []
        for i in range(N):
            zl, = ax_Z.plot([], [], color=colors[i], linewidth=1.0,
                            alpha=0.4, visible=False,
                            label=f"L{i+1}")
            Z_lines.append(zl)
            vl = ax_Z.axvline((i+1)*f0/1e6, color=colors[i],
                              linestyle=':', linewidth=1.0, alpha=0.0)
            harm_vlines_Z.append(vl)

        ax_ratio.set_xlim(f_MHz[0], f_MHz[-1])
        ax_Z.set_xlim(f_MHz[0], f_MHz[-1])
        ax_ratio.set_ylim(0, 1.1)   # will be updated
        ax_Z.set_ylim(1, 1e6)       # will be updated

        self.fig.tight_layout()     # called ONCE here
        leg_ratio = ax_ratio.legend(fontsize=7, ncol=2)
        leg_Z     = ax_Z.legend(fontsize=7, ncol=2)
        self.canvas.draw()

        # Queue for worker -> main-thread plot commands
        q = queue.Queue()

        def worker():
            p = params.copy()
            for k in range(N):
                f_target_k = (k + 1) * f0
                q.put(('status',
                       f"Step {k+1}/{N}: scanning Cs{k+1} for "
                       f"{f_target_k/1e6:.2f} MHz..."))
                p[k, 5] = self._cs_for_max_ratio(
                    p[:k+1], k, f_target_k,
                    Rload, Cload, Lload, Rser, R_div2)

                # ── pre-compute ALL plot data here in the worker thread ──
                snap = p[:k+1].copy()
                ratio_vals = np.array([
                    self._current_ratio_at(f, snap, Rload, Cload, Lload, Rser, R_div2)
                    for f in f_plot])
                Z_curves = [
                    np.array([abs(self.calculate_single_resonator_impedance(f, snap[i]))
                               for f in f_plot])
                    for i in range(len(snap))]
                q.put(('step', k, snap, ratio_vals, Z_curves))

                if k >= 1:
                    q.put(('status',
                           f"Step {k+1}/{N}: joint refinement Cs1..Cs{k+1}..."))
                    self._joint_optimise_cs(p, k + 1, f0,
                                            Rload, Cload, Lload, Rser, R_div2)

                    # Send refined plot after refinement too
                    snap2 = p[:k+1].copy()
                    ratio_vals2 = np.array([
                        self._current_ratio_at(f, snap2, Rload, Cload, Lload, Rser, R_div2)
                        for f in f_plot])
                    Z_curves2 = [
                        np.array([abs(self.calculate_single_resonator_impedance(f, snap2[i]))
                                   for f in f_plot])
                        for i in range(len(snap2))]
                    q.put(('refine', k, snap2, ratio_vals2, Z_curves2))

            q.put(('done', p.copy()))

        # Accumulated ratio curves (one per step) for ghost lines
        ratio_history = []   # list of np arrays

        def _paint(k, snap, ratio_vals, Z_curves, is_refine=False):
            """
            Pure drawing — all numpy arrays already computed in worker thread.
            Only set_data() calls and draw_idle() here, zero computation.
            """
            if is_refine:
                # Update current step's bold line and Z lines in-place
                bold_lines[k].set_data(f_MHz, ratio_vals)
                bold_lines[k].set_label(f"Step {k+1}: +L{k+1} "
                                        f"Cs={snap[k,5]*1e12:.3g}pF ✓")
                ratio_history[k] = ratio_vals   # replace with refined values
            else:
                ratio_history.append(ratio_vals)
                # Demote all previous bold lines to ghosts
                for prev_k in range(k):
                    ghost_lines[prev_k].set_data(f_MHz, ratio_history[prev_k])
                    ghost_lines[prev_k].set_visible(True)
                    bold_lines[prev_k].set_visible(False)

                bold_lines[k].set_data(f_MHz, ratio_vals)
                bold_lines[k].set_label(f"Step {k+1}: +L{k+1} "
                                        f"Cs={snap[k,5]*1e12:.3g}pF")
                bold_lines[k].set_visible(True)

                for i in range(k + 1):
                    harm_vlines_ratio[i].set_alpha(0.55)
                    harm_vlines_Z[i].set_alpha(0.45)

            # Autoscale ratio
            all_vals = np.concatenate(ratio_history)
            ax_ratio.set_ylim(0, max(all_vals.max() * 1.15, 0.1))

            # Update Z lines
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
            self.canvas.draw()
            self.root.update()

        def poll():
            """Drain ONE message per call so tkinter can repaint between steps."""
            try:
                msg = q.get_nowait()

                if msg[0] == 'status':
                    self.status_var.set(msg[1])
                    self.root.update_idletasks()

                elif msg[0] in ('step', 'refine'):
                    _, k, snap, ratio_vals, Z_curves = msg
                    self._update_grid_cs(snap)
                    _paint(k, snap, ratio_vals, Z_curves,
                           is_refine=(msg[0] == 'refine'))
                    # _paint calls canvas.draw() + root.update() — screen updated here

                elif msg[0] == 'done':
                    _, final = msg
                    self._update_grid_cs(final)
                    self._cs_optimized = True
                    try:
                        self._led_stop()
                        self._opt_btn.configure(style="TButton", state="normal")
                    except Exception:
                        pass
                    self.status_var.set(
                        "Cs optimisation complete: " +
                        "  ".join([f"L{i+1}={final[i,5]*1e12:.3g}pF"
                                   for i in range(N)]))
                    return   # stop polling

            except queue.Empty:
                pass

            # Reschedule — 20 ms gives ~50 fps ceiling, feels responsive
            self.root.after(20, poll)

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(50, poll)

    # ═══════════════════════════════════════════════════════════════════════
    # GLOBAL OPTIMIZER  — Genetic Algorithm over inductor database
    # FOM = (E_ref - E_ser) / E_ref * 100  (% reduction in Rser dissipation)
    # ═══════════════════════════════════════════════════════════════════════

    def _compute_fom(self, res_params, f0, mags_in, phases,
                     R_gen_int, R_div1, R_div2, C_line,
                     Z_in_meas, Rser, Rload, Cload, Lload):
        """
        Compute FOM (%) for a given set of resonator params.
        Pure calculation — no GUI interaction.
        Returns float FOM in [0, 100].
        """
        w0 = 2 * np.pi * f0
        n_harm = len(phases)

        # Scale mags so waveform swings 0→1 V (same logic as steady-state plot)
        t_sc = np.linspace(0, 1/f0, 512, endpoint=False)
        V_raw = np.full(512, mags_in[0], dtype=float)
        for i in range(n_harm):
            V_raw += mags_in[i+1] * np.cos((i+1)*w0*t_sc + np.deg2rad(phases[i]))
        swing = np.percentile(V_raw, 98) - np.percentile(V_raw, 2)
        sf = (1.0 / swing) if swing > 1e-12 else 1.0
        mags = [mags_in[0]] + [m * sf for m in mags_in[1:]]

        # Time domain on 10 cycles
        T = 5 / f0
        t = np.linspace(0, T, 2**10 if self._fast_eval else 2**12, endpoint=False)
        dt = t[1] - t[0]
        Iser_t = np.zeros(len(t))
        Iref_t = np.zeros(len(t))

        for n in range(1, n_harm + 1):
            f_h = n * f0
            w_h = 2 * np.pi * f_h
            V_src = mags[n] * np.exp(1j * np.deg2rad(phases[n-1]))
            if abs(V_src) < 1e-12:
                continue

            Y_div2   = 1.0/R_div2 + 1j*w_h*C_line
            Y_shunt  = Y_div2 + 1.0/Z_in_meas
            Z_load_h = Rload + 1j*w_h*Lload + 1.0/(1j*w_h*Cload)

            # With resonators
            Y_res = sum(1.0/self.calculate_single_resonator_impedance(f_h, p)
                        for p in res_params)
            Z_res = 1.0/Y_res if abs(Y_res) > 1e-30 else np.inf
            Z_struct  = 1.0/(1.0/Z_load_h + 1.0/Z_res) if Z_res != np.inf else Z_load_h
            Z_branch  = Rser + Z_struct
            Y_node    = Y_shunt + 1.0/Z_branch
            Z_node    = 1.0/Y_node
            Z_input   = R_div1 + Z_node
            V_node    = V_src * Z_node / (R_gen_int + Z_input)
            Iser_ph   = V_node / Z_branch

            # Reference (no resonators)
            Z_branch_r = Rser + Z_load_h
            Y_node_r   = Y_shunt + 1.0/Z_branch_r
            Z_node_r   = 1.0/Y_node_r
            Z_input_r  = R_div1 + Z_node_r
            V_node_r   = V_src * Z_node_r / (R_gen_int + Z_input_r)
            Iref_ph    = V_node_r / Z_branch_r

            Iser_t += np.abs(Iser_ph) * np.cos(w_h*t + np.angle(Iser_ph))
            Iref_t += np.abs(Iref_ph) * np.cos(w_h*t + np.angle(Iref_ph))

        E_ser = np.sum(Iser_t**2) * dt * Rser
        E_ref = np.sum(Iref_t**2) * dt * Rser
        return (E_ref - E_ser) / E_ref * 100.0 if E_ref > 1e-20 else 0.0

    def _db_row_to_params(self, row, Cs=1e-12, Cp=0.5e-12):
        """Convert a database dict row to [R1,C1,L,k,R_DC,Cs,Cp]."""
        return [
            float(row["R1_Ohm"]),
            float(row["C_pF"]) * 1e-12,
            float(row["L_nH"]) * 1e-9,
            float(row["k"]),
            float(row["R2_Ohm"]),
            Cs,
            Cp,
        ]

    def _optimise_cs_for_individual(self, individual, f0, db_rows,
                                    Rload, Cload, Lload, Rser, R_div2):
        """
        Given a list of N db-row indices, find optimal Cs for each using
        coordinate descent (same as _joint_optimise_cs but builds params inline).
        Returns numpy params array (N,7) with optimised Cs.
        """
        N = len(individual)
        params = np.array([self._db_row_to_params(db_rows[i]) for i in individual])
        # Seed Cs: start from individual Cs scan
        for k in range(N):
            params[k, 5] = self._cs_for_max_ratio(
                params[:k+1], k, (k+1)*f0, Rload, Cload, Lload, Rser, R_div2)
        # Joint coordinate descent
        self._joint_optimise_cs(params, N, f0, Rload, Cload, Lload, Rser, R_div2)
        return params

    def _ga_progress_filename(self, best_fom, names):
        """
        Build a descriptive default filename for the GA progress CSV.
        Format: BP(N)_(N_list)_f=(Fund)MHz_db=(dbname).csv
        e.g.  BP8_16_f=49.5MHz_db=0805CS.csv
        """
        N    = self.ga_n_res.get()
        f0   = self._get_f_fundamental()
        # DB name: stem of loaded file, strip _SPICE_parameters suffix
        db_path = self.inductor_db_path.get()
        db_stem = os.path.splitext(os.path.basename(db_path))[0]
        db_stem = db_stem.replace("_SPICE_parameters", "").replace("_spice_parameters", "")
        # Harmonic count = number of active resonators (Cs > 1e-15)
        n_active = N   # all positions filled by GA
        # Part number prefix — first part name up to first digit run or hyphen
        import re
        pn_prefix = re.sub(r'[-_].*', '', names[0]) if names else "unknown"
        fname = f"BP{N}_{n_active}_f={f0:.1f}MHz_db={db_stem}_FOM{best_fom:.1f}pct.csv"
        return fname

    def _save_ga_progress_prompt(self, gen_best, gen_mean, names, best_fom):
        """Store completed search results for later saving via Save Data menu."""
        self._ga_last_gen_best = gen_best[:]
        self._ga_last_gen_mean = gen_mean[:]
        self._ga_last_names    = names[:]
        self._ga_last_fom      = best_fom
        self.status_var.set(
            f"Search done. Best FOM={best_fom:.2f}%  "
            f"Inductors: {', '.join(names)}  "
            f"— use Files → Save Best Inductors Search to save.")

    def _save_ga_progress(self):
        """Save Best Inductors Search results to CSV (Files menu)."""
        if not getattr(self, '_ga_last_gen_best', None):
            messagebox.showinfo("No Data", "Run a Best Inductors search first.")
            return
        gen_best = self._ga_last_gen_best
        gen_mean = self._ga_last_gen_mean
        names    = self._ga_last_names
        best_fom = self._ga_last_fom
        initial_dir = getattr(self, 'default_save_dir', self.script_dir)
        if not os.path.exists(initial_dir):
            initial_dir = self.script_dir
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=initial_dir,
            initialfile=self._ga_progress_filename(best_fom, names))
        if not filepath:
            return
        try:
            import csv
            with open(filepath, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(["# Best inductors:", ', '.join(names)])
                w.writerow(["# Best FOM (%):", f"{best_fom:.4f}"])
                w.writerow(["# Mode:", self.ga_mode.get()])
                w.writerow(["Iteration", "Best_FOM_pct", "Mean_FOM_pct"])
                for i, (b, m) in enumerate(zip(gen_best, gen_mean)):
                    w.writerow([i, f"{b:.4f}", f"{m:.4f}"])
            self.status_var.set(f"Saved to {os.path.basename(filepath)}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))

    def _stop_ga(self):
        self._ga_running = False
        self.status_var.set("GA: stopping after current evaluation…")

    def _run_ga_optimization(self):
        """
        Genetic algorithm to find the best combination of N inductors
        (from the loaded database) that maximises FOM.

        Chromosome = list of N db-row indices (repeats allowed).
        Fitness    = FOM after Cs optimisation.
        GA ops     = tournament selection, uniform crossover, random mutation.
        """
        if self.inductor_db is None or self.inductor_db.empty:
            messagebox.showwarning("No DB", "Load an inductor database first.")
            return

        try:
            mags, phases, n_harm = self._load_stimulus()
        except Exception as e:
            messagebox.showerror("Stimulus error", str(e))
            return

        N      = self.ga_n_res.get()
        POP    = self.ga_pop.get()
        N_GEN  = self.ga_gen.get()

        f0     = self._get_f_fundamental() * 1e6
        Rload  = self.Rload.get()
        Cload  = self.Cload.get()
        Lload  = self.Lload.get()
        Rser   = self.Rser.get()
        R_div2 = self.R_div2.get()
        R_div1 = self.R_div1.get()
        R_gen  = self.R_internal_gen.get()
        Z_in   = self.Z_in_measure.get()
        C_line = 0.0
        # Rser cell — lets the Rser-optimisation outer loop vary it
        # while ga_worker reads _rser_cell[0]
        _rser_cell = [Rser]

        db_rows = self.inductor_db.to_dict("records")
        DB_SIZE = len(db_rows)

        self._ga_running = True
        self._fast_eval = True
        self._led_start()
        try:
            try:
                self._run_btn.configure(state="disabled")
                self._stop_btn.configure(state="normal")
            except Exception: pass
            try:
                self._optim_run_btn.configure(state="disabled")
                self._optim_stop_btn.configure(state="normal")
                self._optim_rser_btn.configure(state="disabled")
            except Exception: pass
        except Exception: pass
        self.status_var.set("GA: initialising…")
        self.root.update()

        q = queue.Queue()

        def evaluate(individual):
            """
            Optimise Cs for all N resonators, compute FOM.
            No pruning during search — FOM is the sole fitness criterion.
            Returns (fom, params, individual).
            """
            if not individual:
                return 0.0, np.zeros((0, 7)), []
            params = self._optimise_cs_for_individual(
                individual, f0, db_rows, Rload, Cload, Lload, _rser_cell[0], R_div2)
            fom = self._compute_fom(
                params, f0, mags, phases,
                R_gen, R_div1, R_div2, C_line, Z_in,
                _rser_cell[0], Rload, Cload, Lload)
            return fom, params, list(individual)

        mode = self.ga_mode.get()   # "Greedy", "GA only", "Greedy+GA"

        def ga_worker():
          try:
            q.put(('status', 'GA: starting…'))
            rng = np.random.default_rng()
            best_ind = list(rng.integers(0, DB_SIZE, N))
            best_fom = -1.0
            best_par = None
            gen_best = []
            gen_mean = []
            self._ga_progress = gen_best   # live reference for save

            # ══ GREEDY PHASE ══════════════════════════════════════════════
            if mode in ("Greedy", "Greedy+GA"):

                def greedy_cs_pretune(cand_row, k):
                    """
                    Vectorised Cs pre-tune for candidate at position k.
                    Tunes Cs to maximise |Iload/Iser| at harmonic (k+1)*f0.
                    Returns cand_row with Cs set. Cs range: 0.1 pF to 1000 pF.
                    """
                    f_target = (k + 1) * f0
                    w = 2.0 * np.pi * f_target
                    # Search Cs ABOVE series resonance so the branch is inductive
                    # and anti-resonates with Cp to give high impedance at f_target.
                    L_cand = cand_row[0, 2]
                    if L_cand > 0:
                        cs_series_res = 1.0 / (L_cand * (2.0 * np.pi * f_target) ** 2)
                        cs_lo_use  = max(0.01e-12, cs_series_res)
                    else:
                        cs_lo_use  = 0.01e-12
                    cs_max_use = max(1000e-12, cs_lo_use * 10)
                    Cs_arr = np.logspace(np.log10(cs_lo_use), np.log10(cs_max_use), 120)

                    Y_locked = 0j
                    for i in range(len(greedy_params)):
                        Y_locked += 1.0 / self.calculate_single_resonator_impedance(
                            f_target, greedy_params[i])

                    R1, C1, L, kc, R_DC, _, Cp = cand_row[0]
                    ZA   = R1 + 1.0 / (1j * w * C1)
                    ZB   = (kc * np.sqrt(f_target) + R_DC) + 1j * w * L
                    Zp   = 1.0 / (1.0/ZA + 1.0/ZB)
                    Zs   = Zp + 1.0 / (1j * w * Cs_arr)
                    Zcand= 1.0 / (1.0/Zs + 1j * w * Cp)
                    Ytot = Y_locked + 1.0 / Zcand
                    Zres = 1.0 / Ytot
                    Zload= Rload + 1j*w*Lload + 1.0/(1j*w*Cload)
                    Zst  = 1.0 / (1.0/Zload + 1.0/Zres)
                    ratios = np.abs(Zst / Zload)
                    row = cand_row.copy()
                    best_cs = Cs_arr[int(np.argmax(ratios))]
                    best_ratio = ratios[int(np.argmax(ratios))]
                    if best_cs < 0.01e-12 or best_ratio <= 1.0:
                        row[0, 5] = 1e-17   # useless at this harmonic → inactive
                    else:
                        # FIX-6: sub-harmonic reject — anti-resonance below f0
                        L_val, Cp_val = cand_row[0, 2], cand_row[0, 6]
                        if L_val > 0 and Cp_val > 0 and best_cs > 0:
                            f_ar = np.sqrt((best_cs + Cp_val) / (L_val * best_cs * Cp_val)) / (2 * np.pi)
                            if f_ar < f0:
                                row[0, 5] = 1e-17   # sub-harmonic → reject
                                return row
                        row[0, 5] = best_cs
                    return row

                greedy_ind    = []
                greedy_params = np.zeros((0, 7))

                import time as _time
                for k in range(N):
                    if not self._ga_running:
                        if best_par is None:
                            best_par = self._optimise_cs_for_individual(
                                best_ind, f0, db_rows,
                                Rload, Cload, Lload, _rser_cell[0], R_div2)
                        q.put(('stopped', best_ind, best_par, best_fom,
                               db_rows, gen_best[:], gen_mean[:]))
                        return

                    t_scan_start = _time.time()
                    q.put(('status',
                           f"Greedy pos {k+1}/{N}: scanning {DB_SIZE} parts…"))

                    best_k_fom   = -1e9
                    best_k_idx   = 0
                    best_k_par   = None
                    REPORT_EVERY = max(1, DB_SIZE // 10)

                    # Phase 1: fast ratio scan over all DB parts — collect top-K
                    TOP_K = 5   # evaluate top-K by ratio with true FOM
                    ratio_scores = []   # (ratio, di)
                    f_target = (k + 1) * f0
                    for di in range(DB_SIZE):
                        cand_row = np.array([self._db_row_to_params(db_rows[di])])
                        cand_row = greedy_cs_pretune(cand_row, k)
                        active = np.vstack([greedy_params, cand_row])                                  if len(greedy_params) else cand_row
                        ratio = self._current_ratio_at(
                            f_target, active, Rload, Cload, Lload, _rser_cell[0], R_div2)
                        ratio_scores.append((ratio, di))

                        if (di + 1) % REPORT_EVERY == 0 or di == DB_SIZE - 1:
                            elapsed = _time.time() - t_scan_start
                            eta = elapsed / (di + 1) * (DB_SIZE - di - 1)
                            top_now = max(ratio_scores, key=lambda x: x[0])
                            q.put(('progress',
                                   k, di + 1, DB_SIZE, elapsed, eta,
                                   db_rows[top_now[1]]["Part_Number"], top_now[0]))

                    # Phase 2: evaluate top-K by true FOM
                    ratio_scores.sort(key=lambda x: x[0], reverse=True)
                    top_candidates = ratio_scores[:TOP_K]
                    q.put(('status',
                           f"Greedy pos {k+1}/{N}: FOM-scoring top {TOP_K} candidates…"))

                    best_k_fom = -1e9
                    best_k_idx = top_candidates[0][1]
                    best_k_par = None
                    for _, di in top_candidates:
                        trial_ind_k = greedy_ind + [di]
                        fom_di, par_di, _ = evaluate(trial_ind_k)
                        if fom_di > best_k_fom:
                            best_k_fom = fom_di
                            best_k_idx = di
                            best_k_par = par_di

                    # Winner is the top-K candidate with best true FOM
                    trial_ind = greedy_ind + [best_k_idx]
                    fom_k, par_k = best_k_fom, best_k_par

                    if fom_k > best_fom:
                        # Improvement — lock in
                        greedy_ind    = trial_ind
                        greedy_params = par_k
                        best_ind      = greedy_ind[:]
                        best_par      = par_k
                        best_fom      = fom_k
                        q.put(('status',
                               f"Greedy pos {k+1}/{N}: locked "
                               f"{db_rows[best_k_idx]['Part_Number']}"
                               f"  FOM={fom_k:.2f}%"))
                    else:
                        # No improvement — skip this position
                        q.put(('status',
                               f"Greedy pos {k+1}/{N}: no improvement "
                               f"({fom_k:.2f}% vs {best_fom:.2f}%) — skipping"))

                    gen_best.append(best_fom)
                    gen_mean.append(best_fom)
                    q.put(('gen', len(gen_best)-1, gen_best[:], gen_mean[:],
                           best_ind, best_par, best_fom))

                # (Greedy-only exit handled below GA population block)
                if False: pass  # placeholder


            # ══ GA PHASE (pure GA or refinement after greedy) ════════════
            # GA chromosomes are always length N (original requested count).
            # Greedy may have pruned best_ind to shorter length — pad with
            # random genes so all pop entries are exactly length N.
            def pad_to_N(ind):
                """Return a length-N copy, padding with random DB indices if needed."""
                lst = list(ind)
                while len(lst) < N:
                    lst.append(int(rng.integers(0, DB_SIZE)))
                return lst[:N]   # also truncate if somehow longer

            seed_ind = pad_to_N(best_ind)

            # Seed population with greedy result + graduated mutations
            # Slot 0: exact greedy result (preserved as elite)
            # Slots 1..POP//3: light mutations (1 gene changed)
            # Slots POP//3..2*POP//3: medium mutations (2-3 genes)
            # Slots 2*POP//3..: heavy mutations (random)
            pop = [seed_ind[:]]
            third = max(1, POP // 3)
            for j in range(POP - 1):
                mutant = seed_ind[:]
                if j < third:
                    # Light: change exactly 1 gene
                    i = int(rng.integers(0, N))
                    mutant[i] = int(rng.integers(0, DB_SIZE))
                elif j < 2 * third:
                    # Medium: change 2-3 genes
                    for i in rng.choice(N, size=min(3, N), replace=False):
                        mutant[i] = int(rng.integers(0, DB_SIZE))
                else:
                    # Heavy: fully random
                    mutant = list(rng.integers(0, DB_SIZE, N))
                pop.append(mutant)

            # ══ GA PHASE — skip entirely if Greedy-only ═════════════════
            if mode == "Greedy":
                q.put(('done', best_ind, best_par, best_fom, db_rows, gen_best[:], gen_mean[:]))
                return

            fitness = []
            params_cache = []

            # Always evaluate greedy result first (slot 0) to get its exact FOM
            q.put(('status', f"GA init: evaluating greedy seed (1/{POP})…"))
            fom0, par0, pruned0 = evaluate(pop[0])
            fitness.append(fom0)
            params_cache.append(par0)
            # Seed best with whichever is better: greedy FOM or previously stored
            if fom0 > best_fom:
                best_fom = fom0
                best_ind = pruned0   # pruned length — for display/output only
                best_par = par0

            for ii in range(1, POP):
                if not self._ga_running:
                    q.put(('stopped', best_ind, best_par, best_fom, db_rows, gen_best[:], gen_mean[:]))
                    return
                q.put(('status',
                       f"GA init: evaluating individual {ii+1}/{POP}…"))
                fom, params, pruned_ii = evaluate(pop[ii])
                fitness.append(fom)
                params_cache.append(params)
                if fom > best_fom:
                    best_fom = fom
                    best_ind = pruned_ii
                    best_par = params

            gen_best.append(best_fom)
            gen_mean.append(float(np.mean(fitness)))
            q.put(('gen', len(gen_best)-1, gen_best[:], gen_mean[:],
                   best_ind, best_par, best_fom))

            # ── evolve ───────────────────────────────────────────────────
            # Lower mutation rate for Greedy+GA — greedy already found good region
            MUT_RATE  = 0.15 if mode == "Greedy+GA" else 0.25
            ELITE     = max(2, POP // 10)   # top 10% pass unchanged, min 2

            for gen in range(1, N_GEN + 1):
                if not self._ga_running:
                    q.put(('stopped', best_ind, best_par, best_fom, db_rows, gen_best[:], gen_mean[:]))
                    return
                phase_label = "GA refinement" if mode == "Greedy+GA" else "GA"
                q.put(('status',
                       f"{phase_label} gen {gen}/{N_GEN}  best FOM={best_fom:.2f}%…"))

                # Sort by fitness descending
                order = np.argsort(fitness)[::-1]
                pop          = [pop[i]          for i in order]
                fitness      = [fitness[i]      for i in order]
                params_cache = [params_cache[i] for i in order]

                new_pop = pop[:ELITE]   # elitism

                while len(new_pop) < POP:
                    # Tournament selection (size 3)
                    def tournament():
                        cands = rng.integers(0, POP, 3)
                        return pop[int(cands[np.argmax([fitness[c] for c in cands])])]

                    p1, p2 = pad_to_N(tournament()), pad_to_N(tournament())

                    # Uniform crossover
                    child = [p1[i] if rng.random() < 0.5 else p2[i]
                             for i in range(N)]

                    # Mutation
                    for i in range(N):
                        if rng.random() < MUT_RATE:
                            child[i] = int(rng.integers(0, DB_SIZE))

                    new_pop.append(child)

                # Evaluate new members only
                pop = new_pop
                for i in range(ELITE, POP):
                    fom, params, pruned_i = evaluate(pop[i])
                    fitness.append(fom) if i >= len(fitness) else fitness.__setitem__(i, fom)
                    if i >= len(params_cache):
                        params_cache.append(params)
                    else:
                        params_cache[i] = params

                cur_best = int(np.argmax(fitness))
                if fitness[cur_best] > best_fom:
                    best_fom = fitness[cur_best]
                    best_par = params_cache[cur_best]
                    # Get pruned individual for output (cheap — just prune, no full eval)
                    _, _, best_ind = evaluate(pop[cur_best])
                # best_fom is monotonically non-decreasing (global best tracker)
                gen_best.append(best_fom)
                gen_mean.append(float(np.mean(fitness)))

                q.put(('gen', len(gen_best)-1, gen_best[:], gen_mean[:],
                       best_ind, best_par, best_fom))

                # Convergence: stop if best FOM improved by less than 0.1%
                # over the last 5 generations (give GA time to settle first)
                CONVERGE_WINDOW = 5
                CONVERGE_TOL    = 0.1   # percent
                if gen >= CONVERGE_WINDOW:
                    recent_best = gen_best[-CONVERGE_WINDOW]
                    improvement = best_fom - recent_best
                    if improvement < CONVERGE_TOL:
                        q.put(('done', best_ind, best_par, best_fom, db_rows, gen_best[:], gen_mean[:]))
                        return

            q.put(('done', best_ind, best_par, best_fom, db_rows, gen_best[:], gen_mean[:]))
          except Exception as _ex:
            import traceback
            tb = traceback.format_exc()
            print(f"GA WORKER EXCEPTION:\n{tb}")   # always visible in console
            q.put(('status', f"ERROR: {_ex}"))
            q.put(('stopped', best_ind, best_par if best_par is not None else np.zeros((0,7)),
                   best_fom, db_rows, gen_best[:], gen_mean[:]))

        # ── poll ─────────────────────────────────────────────────────────
        # Plot objects created lazily on first 'gen' message (avoids blocking canvas.draw)
        _plot_ready    = [False]
        _best_line_b   = [None]
        _mean_line_g   = [None]
        _bar_cont      = [None]
        _ax_fom        = [None]
        _ax_best       = [None]
        f_harm = np.array([(k+1)*f0 for k in range(N)])

        def _init_plot():
            fig, axes, canvas = self._get_plot_window(
                'optim', 'Best Inductor Search', (8, 6), 2)
            self.fig, self.canvas = fig, canvas
            ax_fom, ax_best = axes[0], axes[1]
            mode_label = {"Greedy": "Greedy scan",
                          "GA only": "GA optimisation",
                          "Greedy+GA": "Greedy + GA refinement",
                          "SA": "Simulated Annealing"}.get(mode, mode)
            ax_fom.set_title(f"{mode_label} — FOM Progress", fontsize=11)
            ax_fom.set_xlabel("Generation / Step", fontsize=11)
            ax_fom.set_ylabel("FOM (%)", fontsize=11)
            ax_fom.grid(True, alpha=0.4)
            ax_best.set_xlabel("Harmonic", fontsize=11)
            ax_best.set_ylabel("|Iload / Iser|", fontsize=11)
            ax_best.set_title("Best Individual — Current Ratio at Each Harmonic", fontsize=11)
            ax_best.grid(True, alpha=0.4)
            bl, = ax_fom.plot([], [], 'b-',  linewidth=2, label="Best FOM")
            mg, = ax_fom.plot([], [], 'g--', linewidth=1, label="Mean FOM")
            ax_fom.legend(fontsize=9)
            harm_x = np.arange(1, N+1)
            bc = ax_best.bar(harm_x, np.zeros(N),
                             color=[plt.cm.tab10(i/max(N,1)) for i in range(N)],
                             alpha=0.75)
            ax_best.set_xticks(harm_x)
            ax_best.set_xticklabels([f"H{i}" for i in harm_x], fontsize=7)
            fig.tight_layout()
            canvas.draw()
            _ax_fom[0]      = ax_fom
            _ax_best[0]     = ax_best
            _best_line_b[0] = bl
            _mean_line_g[0] = mg
            _bar_cont[0]    = bc
            _plot_ready[0]  = True

        def update_plots(gen_best, gen_mean, best_ind, best_par, best_fom):
            if not _plot_ready[0]:
                _init_plot()
            gens = list(range(len(gen_best)))
            _best_line_b[0].set_data(gens, gen_best)
            _mean_line_g[0].set_data(gens, gen_mean)
            _ax_fom[0].set_xlim(-0.5, max(1, len(gens)-1))
            _ax_fom[0].set_ylim(0, max(max(gen_best)*1.1, 1))
            ratios = [self._current_ratio_at(
                          f, best_par, Rload, Cload, Lload, _rser_cell[0], R_div2)
                      for f in f_harm]
            for bar, r in zip(_bar_cont[0], ratios):
                bar.set_height(r)
            _ax_best[0].set_ylim(1, max(max(ratios)*1.15, 1.1))
            _ax_best[0].set_title(
                f"Best FOM={best_fom:.2f}%  "
                f"[{', '.join(db_rows[i]['Part_Number'] for i in best_ind)}]",
                fontsize=11)
            self.canvas.draw()
            self.root.update()

        def poll_ga():
            # Drain ALL pending messages in one shot so status never lags
            done_flag = False
            while True:
                try:
                    msg = q.get_nowait()
                except queue.Empty:
                    break

                if msg[0] == 'status':
                    self.status_var.set(msg[1])

                elif msg[0] == 'progress':
                    _, k, done, total, elapsed, eta, best_name, best_score = msg
                    bar = "█" * (done * 10 // total) + "░" * (10 - done * 10 // total)
                    self.status_var.set(
                        f"Greedy pos {k+1}/{N}  [{bar}] {done}/{total}"
                        f"  {elapsed:.1f}s  ETA {eta:.0f}s"
                        f"  best: {best_name} ({best_score:.3f})")

                elif msg[0] == 'gen':
                    _, gen, gb, gm, bi, bp, bf = msg
                    update_plots(gb, gm, bi, bp, bf)

                elif msg[0] in ('done', 'stopped'):
                    _, best_ind, best_par, best_fom, db_rows_local, gb_final, gm_final = msg
                    if best_par is not None and len(best_par) > 0:
                        self._populate_resonator_grid(best_par, filter_by_ratio=True)
                        self.resonator_params_loaded = best_par.copy()
                    names = [db_rows_local[i]["Part_Number"] for i in best_ind] if best_ind else []
                    self._ga_best_names = names
                    verb = "stopped" if msg[0] == "stopped" else "done"
                    self.status_var.set(
                        f"GA {verb}! Best FOM={best_fom:.2f}%  "
                        f"Inductors: {', '.join(names)}")
                    self._ga_running = False
                    self._fast_eval = False
                    self._led_stop()
                    try:
                        self._run_btn.configure(state="normal")
                        self._stop_btn.configure(state="disabled")
                    except Exception: pass
                    try:
                        self._optim_run_btn.configure(state="normal")
                        self._optim_stop_btn.configure(state="disabled")
                        self._optim_rser_btn.configure(state="normal")
                    except Exception: pass
                    self._save_ga_progress_prompt(gb_final, gm_final, names, best_fom)
                    done_flag = True
                    break

            # Force screen repaint after processing batch
            self.root.update_idletasks()
            if not done_flag:
                self.root.after(50, poll_ga)

        def run_at_rser(rser_val, fast=False):
            """
            Run the full GA/Greedy search with _rser_cell[0]=rser_val.
            Uses a synchronous sub-queue to capture the best FOM without
            interfering with poll_ga.  Messages are forwarded to q so the
            GUI stays live.  Returns best_fom.
            """
            _rser_cell[0] = rser_val
            sub_q = queue.Queue()
            best_fom_out = [0.0]

            import _thread as _thr
            done_evt = threading.Event()

            def sub_worker():
                # Temporarily swap q so ga_worker uses sub_q
                # We can't swap the closure, so we use a flag approach:
                # ga_worker already uses _rser_cell[0] for Rser.
                # Run ga_worker, which puts messages on `q` (outer).
                # We intercept by wrapping: rebuild ga_worker logic with rser_val.
                ga_worker()
                done_evt.set()

            # ga_worker puts done/stopped on outer q — we forward all and
            # capture fom by adding a sentinel after it runs.
            sub_thread = threading.Thread(target=sub_worker, daemon=True)
            sub_thread.start()
            sub_thread.join()   # wait — this is inside outer_worker thread, safe
            # fom will have been put on q as part of done/stopped message
            return 0.0   # fom captured via q by poll_ga

        # ── SA worker ────────────────────────────────────────────────────────
        def sa_worker():
            """
            Simulated Annealing over the inductor database.
            Deterministic (seed=42), single reproducible result.
            Neighbor moves: single-swap (60%), double-swap (30%), random-restart (10%).
            Acceptance: always if better; else with prob exp(ΔFOM / T).
            """
            try:
                T0_sa    = self.sa_t0.get()
                alpha_sa = self.sa_alpha.get()
                iters_sa = self.sa_iters.get()
                rng_sa   = np.random.default_rng(42)

                # ── Seed: best of up to 50 random trials (avoids negative start) ──
                MAX_SEED = 50
                best_seed_fom = -1e9
                best_seed_ind = list(rng_sa.integers(0, DB_SIZE, N))
                best_seed_par = None

                for attempt in range(MAX_SEED):
                    if not self._ga_running:
                        q.put(('stopped', best_seed_ind, best_seed_par or np.zeros((0,7)),
                               best_seed_fom, db_rows, [], []))
                        return
                    q.put(('status',
                           f"SA: seeding {attempt+1}/{MAX_SEED}"
                           f"  best so far={best_seed_fom:.2f}%"))
                    trial_ind = list(rng_sa.integers(0, DB_SIZE, N))
                    trial_fom, trial_par, _ = evaluate(trial_ind)
                    if trial_fom > best_seed_fom:
                        best_seed_fom = trial_fom
                        best_seed_ind = trial_ind[:]
                        best_seed_par = trial_par
                    # keep trying all attempts to find the best seed

                current_ind = best_seed_ind[:]
                current_fom = best_seed_fom
                current_par = best_seed_par

                best_ind = current_ind[:]
                best_fom = current_fom
                best_par = current_par

                T = T0_sa
                fom_history = [best_fom]
                accept_history = [best_fom]   # tracks current (not best) for mean line

                q.put(('status', f"SA start  FOM={best_fom:.2f}%  T={T:.3f}"))
                q.put(('gen', 0, fom_history[:], accept_history[:],
                       best_ind, best_par, best_fom))

                CONVERGE_WINDOW = 80   # steps
                CONVERGE_TOL   = 0.01  # % — stop if best FOM gains less than this

                for step in range(1, iters_sa + 1):
                    if not self._ga_running:
                        q.put(('stopped', best_ind, best_par, best_fom,
                               db_rows, fom_history[:], accept_history[:]))
                        return

                    # ── Generate neighbour ───────────────────────────────────
                    roll = rng_sa.random()
                    neighbor = current_ind[:]
                    if roll < 0.60 or N == 1:
                        # Single swap
                        i = int(rng_sa.integers(0, N))
                        neighbor[i] = int(rng_sa.integers(0, DB_SIZE))
                    elif roll < 0.90:
                        # Double swap
                        idxs = rng_sa.choice(N, size=min(2, N), replace=False)
                        for i in idxs:
                            neighbor[i] = int(rng_sa.integers(0, DB_SIZE))
                    else:
                        # Full random restart — helps escape deep basins
                        neighbor = list(rng_sa.integers(0, DB_SIZE, N))

                    neighbor_fom, neighbor_par, _ = evaluate(neighbor)
                    delta = neighbor_fom - current_fom

                    # ── SA acceptance criterion ──────────────────────────────
                    if delta > 0 or rng_sa.random() < np.exp(delta / max(T, 1e-9)):
                        current_ind = neighbor
                        current_fom = neighbor_fom
                        current_par = neighbor_par

                    if current_fom > best_fom:
                        best_fom = current_fom
                        best_ind = current_ind[:]
                        best_par = current_par

                    T *= alpha_sa
                    fom_history.append(best_fom)
                    accept_history.append(current_fom)

                    if step % 5 == 0 or step == iters_sa:
                        q.put(('status',
                               f"SA step {step}/{iters_sa}  T={T:.4f}"
                               f"  current={current_fom:.2f}%"
                               f"  best={best_fom:.2f}%"))
                        q.put(('gen', step, fom_history[:], accept_history[:],
                               best_ind, best_par, best_fom))

                    # ── Convergence check ────────────────────────────────────
                    if step >= CONVERGE_WINDOW:
                        improvement = best_fom - fom_history[-CONVERGE_WINDOW]
                        if improvement < CONVERGE_TOL:
                            q.put(('status',
                                   f"SA converged at step {step}  "
                                   f"(ΔFOM={improvement:.4f}% < {CONVERGE_TOL}%)"
                                   f"  best={best_fom:.2f}%"))
                            q.put(('gen', step, fom_history[:], accept_history[:],
                                   best_ind, best_par, best_fom))
                            break

                q.put(('done', best_ind, best_par, best_fom,
                       db_rows, fom_history[:], accept_history[:]))

            except Exception as _ex:
                import traceback
                print(f"SA WORKER EXCEPTION:\n{traceback.format_exc()}")
                q.put(('status', f"SA ERROR: {_ex}"))
                q.put(('stopped',
                       best_ind if 'best_ind' in dir() else [],
                       best_par if 'best_par' in dir() else np.zeros((0, 7)),
                       best_fom if 'best_fom' in dir() else 0.0,
                       db_rows, [], []))

        def outer_worker():
            """Dispatch to SA or GA/Greedy based on current mode setting."""
            if self.ga_mode.get() == "SA":
                sa_worker()
            else:
                ga_worker()
        threading.Thread(target=outer_worker, daemon=True).start()
        self.root.after(20, poll_ga)

    def _run_rser_optimization(self):
        """
        Post-search Rser optimisation: fix the current resonator params from the
        GUI, sweep Rser via golden-section (re-optimising Cs only at each point),
        write best Rser and updated Cs back to the panel.
        Runs in a background thread so the GUI stays responsive.
        """
        params = self._get_resonator_params_from_gui()
        if len(params) == 0:
            from tkinter import messagebox
            messagebox.showwarning("No resonators", "Run Find Best Inductors first.")
            return

        rser_lo = self.ga_rser_min.get()
        rser_hi = self.ga_rser_max.get()
        if rser_lo >= rser_hi or rser_lo <= 0:
            from tkinter import messagebox
            messagebox.showwarning("Invalid range",
                                   f"Rser range {rser_lo}–{rser_hi} Ω is invalid.")
            return

        try:
            mags, phases, _ = self._load_stimulus()
        except Exception as e:
            from tkinter import messagebox
            messagebox.showerror("Stimulus error", str(e))
            return

        f0     = self._get_f_fundamental() * 1e6
        Rload  = self.Rload.get()
        Cload  = self.Cload.get()
        Lload  = self.Lload.get()
        R_div2 = self.R_div2.get()
        R_div1 = self.R_div1.get()
        R_gen  = self.R_internal_gen.get()
        Z_in   = self.Z_in_measure.get()
        C_line = 0.0
        fixed_par = params.copy()

        self._led_start()
        try:
            self._optim_rser_btn.configure(state="disabled")
            self._optim_run_btn.configure(state="disabled")
        except Exception: pass
        self.status_var.set(f"Rser opt: sweeping {rser_lo:.0f}–{rser_hi:.0f} Ω…")
        self.root.update_idletasks()

        def worker():
            PHI = (np.sqrt(5) - 1) / 2   # 0.618…
            GS_ITERS = 10                 # function evaluations

            def fom_at(rser_val):
                # Tweak Cs via coordinate descent from current values (reseed=False)
                # so the optimised Cs are preserved and only slightly adjusted
                # for the new Rser — avoids full redistribution.
                p = fixed_par.copy()
                self._joint_optimise_cs(
                    p, len(p), f0, Rload, Cload, Lload, rser_val, R_div2,
                    reseed=False)
                fom = self._compute_fom(
                    p, f0, mags, phases,
                    R_gen, R_div1, R_div2, C_line, Z_in,
                    rser_val, Rload, Cload, Lload)
                return fom, p

            a, b = rser_lo, rser_hi
            c = b - PHI * (b - a)
            d = a + PHI * (b - a)
            fc, _ = fom_at(c)
            fd, _ = fom_at(d)

            for i in range(GS_ITERS - 2):
                self.root.after(0, lambda rv=((a+b)/2), i=i: self.status_var.set(
                    f"Rser opt: step {i+3}/{GS_ITERS}, testing ≈{rv:.1f} Ω…"))
                if fc > fd:
                    # maximum is in [a, d] — shrink from right
                    b = d
                    d, fd = c, fc
                    c = b - PHI * (b - a)
                    fc, _ = fom_at(c)
                else:
                    # maximum is in [c, b] — shrink from left
                    a = c
                    c, fc = d, fd
                    d = a + PHI * (b - a)
                    fd, _ = fom_at(d)

            best_rser = (a + b) / 2.0
            best_fom, p_final = fom_at(best_rser)
            self.root.after(0, lambda: _done(best_rser, best_fom, p_final))

        def _done(best_rser, best_fom, p_final):
            self.Rser.set(round(best_rser, 2))
            self._update_grid_cs(p_final)
            self._cs_optimized = True
            self._led_stop()
            self.status_var.set(
                f"Rser opt done: best Rser={best_rser:.1f} Ω  FOM={best_fom:.2f}%"
                f"  (Cs tweaked)")
            try:
                self._optim_rser_btn.configure(state="normal")
                self._optim_run_btn.configure(state="normal")
            except Exception: pass

        import threading as _thr
        _thr.Thread(target=worker, daemon=True).start()

    def _led_start(self):
        """Start flashing the activity LED."""
        self._led_on = True
        self._led_flash()

    def _led_stop(self):
        """Stop flashing and turn LED grey (idle)."""
        self._led_on = False
        if self._led_job:
            self.root.after_cancel(self._led_job)
            self._led_job = None
        try:
            self._led_canvas.itemconfig(self._led, fill="#cccccc")
        except Exception:
            pass

    def _led_flash(self):
        """Toggle LED colour and reschedule."""
        if not self._led_on:
            return
        try:
            cur = self._led_canvas.itemcget(self._led, 'fill')
            self._led_canvas.itemconfig(
                self._led, fill="#00cc44" if cur != "#00cc44" else "#005522")
        except Exception:
            pass
        self._led_job = self.root.after(400, self._led_flash)

    def _open_optim_window(self):
        """Open (or show) the Best Inductors search window without starting a search.
        The control strip with all parameters and the Run button lives there."""
        if self.ref_calc_var.get():
            messagebox.showinfo("Reference Mode",
                                "Disable Reference Calculation mode before running Best Inductors.")
            return
        key = 'optim'
        win = self._plot_wins.get(key)
        if win is None or not win.winfo_exists():
            # Create the window with control strip but no plot yet
            win = tk.Toplevel(self.root)
            win.title("Best Inductor Search")
            win.geometry("860x620")
            win.resizable(True, True)
            win.protocol("WM_DELETE_WINDOW", win.withdraw)

            ctrl = ttk.Frame(win, padding="6 4 6 4")
            ctrl.pack(side=tk.TOP, fill=tk.X)
            self._build_plot_controls(key, ctrl)
            ttk.Separator(win, orient="horizontal").pack(fill=tk.X)

            # Placeholder canvas — will be replaced when search runs
            fig = plt.Figure(figsize=(10, 6))
            canvas = FigureCanvasTkAgg(fig, master=win)
            toolbar = NavigationToolbar2Tk(canvas, win)
            toolbar.update()
            toolbar.pack(side=tk.BOTTOM, fill=tk.X)
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            self._plot_wins[key]   = win
            self._plot_figs[key]   = fig
            self._plot_canvas[key] = canvas
        else:
            win.deiconify()
            win.lift()

    def _get_plot_window(self, key, title, figsize, nrows):
        """
        Return (fig, axes, canvas) for the given plot window key.
        Creates a new Toplevel on first call or if the window was closed.
        The per-window control strip (ctrl_frame) is built once by
        _build_plot_controls() and never rebuilt.
        """
        win = self._plot_wins.get(key)
        if win is None or not win.winfo_exists():
            win = tk.Toplevel(self.root)
            win.title(title)
            win.geometry(f"{int(figsize[0]*95)}x{int(figsize[1]*95)+60}")
            win.resizable(True, True)
            win.protocol("WM_DELETE_WINDOW", win.withdraw)

            # ── control strip at top ──────────────────────────────────────
            ctrl = ttk.Frame(win, padding="4 2 4 2")
            ctrl.pack(side=tk.TOP, fill=tk.X)
            self._build_plot_controls(key, ctrl)
            ttk.Separator(win, orient="horizontal").pack(fill=tk.X)

            # ── matplotlib canvas ─────────────────────────────────────────
            fig = plt.Figure(figsize=figsize)
            canvas = FigureCanvasTkAgg(fig, master=win)
            toolbar = NavigationToolbar2Tk(canvas, win)
            toolbar.update()
            toolbar.pack(side=tk.BOTTOM, fill=tk.X)
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            self._plot_wins[key]   = win
            self._plot_figs[key]   = fig
            self._plot_canvas[key] = canvas
        else:
            win.deiconify()
            win.lift()
            fig    = self._plot_figs[key]
            canvas = self._plot_canvas[key]

        fig.clf()
        axes = fig.subplots(nrows, 1)
        if nrows == 1:
            axes = [axes]
        return fig, axes, canvas

    def _build_plot_controls(self, key, parent):
        """Build the control strip inside a plot window (called once at creation)."""
        if key == 'freq':
            # Start / End freq
            ttk.Label(parent, text="Start (MHz):").pack(side=tk.LEFT)
            ttk.Entry(parent, textvariable=self.f_min,  width=7).pack(side=tk.LEFT, padx=(2,8))
            ttk.Label(parent, text="End (MHz):").pack(side=tk.LEFT)
            ttk.Entry(parent, textvariable=self.f_max,  width=7).pack(side=tk.LEFT, padx=(2,12))
            # V_node / S-param switch
            ttk.Radiobutton(parent, text="V node",
                            variable=self.freq_plot_type, value="V_node"
                            ).pack(side=tk.LEFT, padx=2)
            ttk.Radiobutton(parent, text="S-param (dB)",
                            variable=self.freq_plot_type, value="S21"
                            ).pack(side=tk.LEFT, padx=2)
            ttk.Combobox(parent, textvariable=self.s_param_choice,
                         values=["S11", "S21", "Both"], state="readonly",
                         width=6).pack(side=tk.LEFT, padx=(2,12))
            # Subtract reference
            ttk.Checkbutton(parent, text="Subtract Reference",
                            variable=self.subtract_ref_var).pack(side=tk.LEFT, padx=4)
            def _run_freq():
                self._run_frequency()
            ttk.Button(parent, text="▶ Run", command=_run_freq).pack(side=tk.RIGHT, padx=4)

        elif key == 'steady':
            ttk.Checkbutton(parent, text="Subtract Reference",
                            variable=self.subtract_ref_var).pack(side=tk.LEFT, padx=4)
            ttk.Checkbutton(parent, text="Show V_gen",
                            variable=self.show_vgen_var).pack(side=tk.LEFT, padx=4)
            ttk.Button(parent, text="▶ Run",
                       command=self._run_steady_state).pack(side=tk.RIGHT, padx=4)

        elif key == 'cs_opt':
            ttk.Button(parent, text="▶ Run Cs Opt",
                       command=self._run_cs_optimization).pack(side=tk.RIGHT, padx=4)

        elif key == 'optim':
            p = parent
            r0 = ttk.Frame(p); r0.pack(fill=tk.X, pady=1)
            ttk.Label(r0, text="Mode:", width=8, anchor="w").pack(side=tk.LEFT)
            for m in ("Greedy", "GA only", "Greedy+GA", "SA"):
                ttk.Radiobutton(r0, text=m, variable=self.ga_mode,
                                value=m).pack(side=tk.LEFT, padx=(0,4))

            r1 = ttk.Frame(p); r1.pack(fill=tk.X, pady=1)
            ttk.Label(r1, text="N inductors:", width=10, anchor="w").pack(side=tk.LEFT)
            ttk.Combobox(r1, textvariable=self.ga_n_res,
                         values=list(range(1, 13)), state="readonly",
                         width=4).pack(side=tk.LEFT, padx=(0,12))
            ttk.Label(r1, text="Pop:").pack(side=tk.LEFT)
            self._pop_combobox = ttk.Combobox(r1, textvariable=self.ga_pop,
                         values=[10,20,30,50], state="readonly", width=5)
            self._pop_combobox.pack(side=tk.LEFT, padx=(2,8))
            ttk.Label(r1, text="Gen:").pack(side=tk.LEFT)
            ttk.Combobox(r1, textvariable=self.ga_gen,
                         values=[10,20,40,80], state="readonly",
                         width=4).pack(side=tk.LEFT, padx=(2,0))

            r2 = ttk.Frame(p); r2.pack(fill=tk.X, pady=1)
            ttk.Label(r2, text="Keep if ΔFOM ≥", anchor="w").pack(side=tk.LEFT)
            ttk.Entry(r2, textvariable=self.ga_min_contrib,
                      width=5).pack(side=tk.LEFT, padx=(2,2))
            ttk.Label(r2, text="%  (leave-one-out; 0 = keep all)").pack(side=tk.LEFT)

            r3 = ttk.Frame(p); r3.pack(fill=tk.X, pady=1)
            ttk.Label(r3, text="range (Ω):").pack(side=tk.LEFT)
            ttk.Entry(r3, textvariable=self.ga_rser_min,
                      width=6).pack(side=tk.LEFT, padx=(2,1))
            ttk.Label(r3, text="–").pack(side=tk.LEFT)
            ttk.Entry(r3, textvariable=self.ga_rser_max,
                      width=6).pack(side=tk.LEFT, padx=(1,0))

            # SA-specific parameters row
            r3b = ttk.Frame(p); r3b.pack(fill=tk.X, pady=1)
            ttk.Label(r3b, text="SA:  T₀", anchor="w").pack(side=tk.LEFT)
            ttk.Entry(r3b, textvariable=self.sa_t0,    width=5).pack(side=tk.LEFT, padx=(2,6))
            ttk.Label(r3b, text="α").pack(side=tk.LEFT)
            ttk.Entry(r3b, textvariable=self.sa_alpha, width=5).pack(side=tk.LEFT, padx=(2,6))
            ttk.Label(r3b, text="Steps").pack(side=tk.LEFT)
            ttk.Entry(r3b, textvariable=self.sa_iters, width=5).pack(side=tk.LEFT, padx=(2,0))
            ttk.Label(r3b, text="  (used only when mode=SA)",
                      foreground="grey", font=("TkDefaultFont", 8)).pack(side=tk.LEFT, padx=(6,0))

            r4 = ttk.Frame(p); r4.pack(fill=tk.X, pady=(6,2))
            def _run_search():
                self.analysis_choice.set("Best Inductors")
                self._run_ga_optimization()
            self._optim_run_btn = ttk.Button(r4, text="▶ Find Best Inductors",
                                              command=_run_search)
            self._optim_run_btn.pack(side=tk.LEFT, padx=(0,6))
            self._optim_stop_btn = ttk.Button(r4, text="■ Stop",
                                               command=self._stop_ga, state="disabled")
            self._optim_stop_btn.pack(side=tk.LEFT, padx=(0,12))
            ttk.Separator(r4, orient="vertical").pack(side=tk.LEFT, fill="y", padx=6)
            self._optim_rser_btn = ttk.Button(r4, text="⚙ Optimize Rser",
                                               command=self._run_rser_optimization,
                                               state="disabled")
            self._optim_rser_btn.pack(side=tk.LEFT, padx=(0,4))
            ttk.Label(r4, text="(run after Find Best Inductors)",
                      foreground="grey", font=("TkDefaultFont", 8)).pack(side=tk.LEFT)

    def _open_and_run(self, mode):
        """Open the appropriate analysis window and immediately run it."""
        if mode == "Frequency":
            self._run_frequency()
        elif mode == "Steady State":
            self._run_steady_state()
        elif mode == "Cs Opt":
            self._run_cs_optimization()

    def _run_frequency(self):
        """Run frequency analysis (opens/shows freq window then runs)."""
        self.analysis_choice.set("Frequency")
        self.run_analysis()

    def _run_steady_state(self):
        """Run steady state analysis."""
        self.analysis_choice.set("Steady State")
        self.run_analysis()

    def _update_run_button(self, *_):
        """Keep Run button label and state in sync with analysis_choice."""
        choice = self.analysis_choice.get()
        labels = {
            "Frequency":      "Run Freq. Analysis",
            "Steady State":   "Run Steady State",
            "Cs Opt":         "Optimize Cs",
            "Best Inductors": "Open Search Window",
        }
        self._run_btn.configure(text=labels.get(choice, "Run"))
        # Stop button only relevant for Best Inductors
        if not self._ga_running:
            self._stop_btn.configure(
                state="normal" if choice == "Best Inductors" and self._ga_running
                else "disabled")

    def _run_dispatch(self):
        """Legacy — kept for compat. Use Analysis menu instead."""
        pass

    def run_analysis(self):
        if self.analysis_choice.get() not in ("Frequency", "Steady State"):
            return   # guarded: dispatch handles Cs Opt / Best Inductors
        self.status_var.set("Running analysis...")
        self.root.update_idletasks()
        try:
            mags, phases, n_harm = self._load_stimulus()
            res_params = self._get_resonator_params_from_gui()
            
            v_rms_mv = self.V_gen_rms.get()
            R_gen_int = self.R_internal_gen.get()
            
            if self.gen_matched_load.get():
                v_ideal_rms = v_rms_mv * 2.0
            else:
                v_ideal_rms = v_rms_mv
            
            R_div1, R_div2, C_line = self.R_div1.get(), self.R_div2.get(), 0.0  # C_line fixed = 0
            Z_in_meas = self.Z_in_measure.get() 
            
            Rser = self.Rser.get() 
            Rload, Cload, Lload = self.Rload.get(), self.Cload.get(), self.Lload.get()
            
            if self.analysis_choice.get() == "Frequency":
                fig, axes, canvas = self._get_plot_window(
                    'freq', 'Frequency Analysis', (8, 10), 5)
                self.fig, self.axes, self.canvas = fig, axes, canvas
                self._run_frequency_analysis(v_ideal_rms, R_gen_int, R_div1, R_div2, C_line, Z_in_meas, Rser, Rload, Cload, Lload, res_params)
            else:
                fig, axes, canvas = self._get_plot_window(
                    'steady', 'Steady State Analysis', (8, 9), 4)
                self.fig, self.axes, self.canvas = fig, axes, canvas
                self._run_steady_state_analysis(v_ideal_rms, R_gen_int, R_div1, R_div2, C_line, Z_in_meas, Rser, Rload, Cload, Lload, res_params, mags, phases, n_harm)

            fig.subplots_adjust(
                left=0.10, right=0.97,
                top=0.96, bottom=0.06,
                hspace=0.55)
            canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", str(e)); self.status_var.set(f"Error: {e}")

    def calculate_single_resonator_impedance(self, f, params):
        R1, C1, L, k, R_DC, Cs, Cp = params
        if f < 1e-6: return np.inf
        if Cs < 1e-16:  return np.inf   # Cs < 1e-5 pF → open circuit, resonator inactive
        omega = 2 * np.pi * f
        Z_A = R1 + 1 / (1j * omega * C1)
        R_se_dynamic = k * np.sqrt(f)
        Z_B = (R_se_dynamic + R_DC) + 1j * omega * L
        Z_AB_parallel = 1 / (1/Z_A + 1/Z_B)
        Z_series_block = Z_AB_parallel + 1 / (1j * omega * Cs)
        return 1 / (1/Z_series_block + 1j * omega * Cp)

    # --------------------------------------------------------------------------------
    # FREQUENCY ANALYSIS
    # --------------------------------------------------------------------------------
    def _run_frequency_analysis(self, v_rms_mv, R_gen_int, R_div1, R_div2, C_line, Z_in_meas, Rser, Rload, Cload, Lload, res_params):
        f_min, f_max = self.f_min.get() * 1e6, self.f_max.get() * 1e6
        freqs = np.linspace(f_min, f_max, 2000)
        
        # Generator Definition
        V_peak_source = (v_rms_mv * 1e-3) * np.sqrt(2)
        V_source_ideal = V_peak_source + 0j
        
        # Containers
        Iser_list, Iload_list, V_node_list = [], [], []
        s21_list, s11_list = [], []
        
        # Reference Containers
        Iser_ref_list, Iload_ref_list, V_node_ref_list = [], [], []
        s21_ref_list, s11_ref_list = [], []

        for f in freqs:
            w = 2*np.pi*f
            
            # --- 1. Downstream Circuit (The "Device" shunting the node) ---
            Y_res = sum(1/self.calculate_single_resonator_impedance(f, p) for p in res_params)
            Z_res_total = 1.0/Y_res if Y_res!=0 else np.inf
            
            Z_load_comp = Rload + 1j*w*Lload + (1/(1j*w*Cload) if w>0 else np.inf)
            Z_struct = 1.0 / (1.0/Z_load_comp + 1.0/Z_res_total) if Z_res_total != np.inf else Z_load_comp
            Z_branch = Rser + Z_struct
            
            if w == 0: Y_div2_part = 1.0/R_div2
            else: Y_div2_part = 1.0/R_div2 + 1j*w*C_line
            
            Y_device_shunt = Y_div2_part + (1.0/Z_branch)
            Z_device_shunt = 1.0/Y_device_shunt
            
            # --- 2. Solve for V_node (Linear) using Port 2 Termination ---
            Z_port2_load = Z_in_meas
            Z_total_node_load = 1.0 / (Y_device_shunt + 1.0/Z_port2_load)
            
            Z_input_seen_from_src = R_div1 + Z_total_node_load
            V_node_complex = V_source_ideal * (Z_total_node_load / (R_gen_int + Z_input_seen_from_src))
            
            # --- 3. Calculate Currents ---
            Iser = V_node_complex / Z_branch
            V_load_voltage = Iser * Z_struct
            Iload = V_load_voltage / Z_load_comp
            
            Iser_list.append(Iser)
            Iload_list.append(Iload)
            V_node_list.append(V_node_complex)
            
            # --- 4. Calculate S-Parameters ---
            s21_val = 2.0 * V_node_complex / V_source_ideal
            if R_gen_int != Z_in_meas:
                 s21_val = s21_val * np.sqrt(R_gen_int / Z_in_meas)
            s21_list.append(s21_val)
            
            Z_in_total = Z_input_seen_from_src
            s11_val = (Z_in_total - R_gen_int) / (Z_in_total + R_gen_int)
            s11_list.append(s11_val)

            # --- 5. Reference Circuit (Without Resonators) ---
            Z_struct_ref = Z_load_comp
            Z_branch_ref = Rser + Z_struct_ref
            Y_device_shunt_ref = Y_div2_part + (1.0/Z_branch_ref)
            Z_total_node_load_ref = 1.0 / (Y_device_shunt_ref + 1.0/Z_port2_load)
            Z_input_seen_from_src_ref = R_div1 + Z_total_node_load_ref
            
            V_node_complex_ref = V_source_ideal * (Z_total_node_load_ref / (R_gen_int + Z_input_seen_from_src_ref))
            Iser_ref = V_node_complex_ref / Z_branch_ref
            V_load_voltage_ref = Iser_ref * Z_struct_ref
            Iload_ref = V_load_voltage_ref / Z_load_comp
            
            s21_val_ref = 2.0 * V_node_complex_ref / V_source_ideal
            if R_gen_int != Z_in_meas:
                s21_val_ref = s21_val_ref * np.sqrt(R_gen_int / Z_in_meas)
            s11_val_ref = (Z_input_seen_from_src_ref - R_gen_int) / (Z_input_seen_from_src_ref + R_gen_int)
            
            Iser_ref_list.append(Iser_ref)
            Iload_ref_list.append(Iload_ref)
            V_node_ref_list.append(V_node_complex_ref)
            s21_ref_list.append(s21_val_ref)
            s11_ref_list.append(s11_val_ref)

        # Conversion
        Iser_arr = np.array(Iser_list); Iser_ref_arr = np.array(Iser_ref_list)
        Iload_arr = np.array(Iload_list); Iload_ref_arr = np.array(Iload_ref_list)
        V_node_arr = np.array(V_node_list); V_node_ref_arr = np.array(V_node_ref_list)
        s21_arr = np.array(s21_list); s21_ref_arr = np.array(s21_ref_list)
        s11_arr = np.array(s11_list); s11_ref_arr = np.array(s11_ref_list)
        
        s21_db = 20 * np.log10(np.abs(s21_arr) + 1e-12)
        s11_db = 20 * np.log10(np.abs(s11_arr) + 1e-12)
        V_node_rms = np.abs(V_node_arr) / np.sqrt(2)

        s21_ref_db = 20 * np.log10(np.abs(s21_ref_arr) + 1e-12)
        s11_ref_db = 20 * np.log10(np.abs(s11_ref_arr) + 1e-12)
        V_node_ref_rms = np.abs(V_node_ref_arr) / np.sqrt(2)

        # Apply Reference Subtraction 
        if self.subtract_ref_var.get():
            plot_Iser = np.abs(Iser_arr) - np.abs(Iser_ref_arr)
            plot_Iload = np.abs(Iload_arr) - np.abs(Iload_ref_arr)
            plot_Iser_ang = np.angle(Iser_arr, deg=True) - np.angle(Iser_ref_arr, deg=True)
            plot_Iload_ang = np.angle(Iload_arr, deg=True) - np.angle(Iload_ref_arr, deg=True)
            plot_ratio = (np.abs(Iload_arr)/np.abs(Iser_arr)) - (np.abs(Iload_ref_arr)/np.abs(Iser_ref_arr))
            
            plot_s21_db = s21_db - s21_ref_db
            plot_s11_db = s11_db - s11_ref_db
            plot_s21_ang = np.angle(s21_arr, deg=True) - np.angle(s21_ref_arr, deg=True)
            plot_s11_ang = np.angle(s11_arr, deg=True) - np.angle(s11_ref_arr, deg=True) # NEW S11 Phase Diff
            
            plot_V_node_rms = V_node_rms - V_node_ref_rms
            plot_V_node_ang = np.angle(V_node_arr, deg=True) - np.angle(V_node_ref_arr, deg=True)
            y_mod = " (Delta)"
        else:
            plot_Iser = np.abs(Iser_arr)
            plot_Iload = np.abs(Iload_arr)
            plot_Iser_ang = np.angle(Iser_arr, deg=True)
            plot_Iload_ang = np.angle(Iload_arr, deg=True)
            plot_ratio = np.abs(Iload_arr)/np.abs(Iser_arr)
            
            plot_s21_db = s21_db
            plot_s11_db = s11_db
            plot_s21_ang = np.angle(s21_arr, deg=True)
            plot_s11_ang = np.angle(s11_arr, deg=True) # NEW S11 Phase Normal
            
            plot_V_node_rms = V_node_rms
            plot_V_node_ang = np.angle(V_node_arr, deg=True)
            y_mod = ""

        # Plotting
        self.axes[0].plot(freqs/1e6, plot_Iser, label='I_ser' + y_mod)
        self.axes[0].plot(freqs/1e6, plot_Iload, label='I_load' + y_mod)
        if not self.subtract_ref_var.get():
            self.axes[0].set_yscale('log') # Disable log scale if plotting diff/deltas (which can be negative)
        self.axes[0].legend(fontsize=7, loc='best'); self.axes[0].grid(True)
        self.axes[0].set_ylabel(f'Current (A){y_mod}')
        
        self.axes[1].plot(freqs/1e6, plot_Iser_ang, label='I_ser' + y_mod)
        self.axes[1].plot(freqs/1e6, plot_Iload_ang, label='I_load' + y_mod)
        self.axes[1].legend(fontsize=7, loc='best'); self.axes[1].grid(True)
        self.axes[1].set_ylabel(f'Phase (deg){y_mod}')
        
        self.axes[2].plot(freqs/1e6, plot_ratio, color='purple')
        self.axes[2].set_ylabel(f"Current Ratio{y_mod}"); self.axes[2].grid(True)

        plot_mode = self.freq_plot_type.get()
        if plot_mode == "S21":
            s_choice = self.s_param_choice.get() # Grab the new dropdown value
            
            # Conditionally plot magnitude and phase
            if s_choice in ["S21", "Both"]:
                self.axes[3].plot(freqs/1e6, plot_s21_db, color='blue', label='S21 (dB)' + y_mod)
                self.axes[4].plot(freqs/1e6, plot_s21_ang, color='blue', label='Ang(S21)' + y_mod)
                
            if s_choice in ["S11", "Both"]:
                self.axes[3].plot(freqs/1e6, plot_s11_db, color='green', linestyle='--', label='S11 (dB)' + y_mod)
                self.axes[4].plot(freqs/1e6, plot_s11_ang, color='green', linestyle='--', label='Ang(S11)' + y_mod)

            self.axes[3].set_ylabel(f"Magnitude (dB){y_mod}"); self.axes[3].grid(True); self.axes[3].legend(fontsize=7, loc='best')
            self.axes[4].set_ylabel(f"Phase (deg){y_mod}"); self.axes[4].set_xlabel("Frequency (MHz)"); self.axes[4].grid(True)
            self.axes[4].legend(fontsize=7, loc='best')
        else:
            self.axes[3].plot(freqs/1e6, plot_V_node_rms, color='orange', label='|V_node| (RMS)' + y_mod)
            self.axes[3].set_ylabel(f"|V_node| (V RMS){y_mod}"); self.axes[3].grid(True); self.axes[3].legend(fontsize=7, loc='best')
            
            self.axes[4].plot(freqs/1e6, plot_V_node_ang, color='brown', label='Phase(V_node)' + y_mod)
            self.axes[4].set_ylabel(f"Phase (deg){y_mod}"); self.axes[4].set_xlabel("Frequency (MHz)"); self.axes[4].grid(True)
            self.axes[4].legend(fontsize=7, loc='best')
        
        # Harmonics
        f_fund_val = self._get_f_fundamental()
        f_start = self.f_min.get()
        f_end = self.f_max.get()
        n = 1
        current_harmonic = f_fund_val * n
        while current_harmonic <= f_end:
            if current_harmonic >= f_start:
                for ax in self.axes:
                    ax.axvline(x=current_harmonic, color='r', linestyle=':', alpha=0.5, linewidth=1.5)
                self.axes[0].text(current_harmonic, self.axes[0].get_ylim()[1], f"{n}f", 
                                  color='r', ha='center', va='bottom', fontsize=8, rotation=0)
            n += 1
            current_harmonic = f_fund_val * n

        self.last_filename = self._build_save_filename(
            'freq', self._get_f_fundamental() * 1e6, res_params)
        data = {
            'Frequency_MHz':    freqs / 1e6,
            'I_ser_RMS':        plot_Iser,
            'Phase_I_ser_deg':  plot_Iser_ang,
            'I_load_RMS':       plot_Iload,
            'Phase_I_load_deg': plot_Iload_ang,
            'Current_Ratio':    plot_ratio,
            'V_node_RMS':       plot_V_node_rms,
            'Phase_V_node_deg': plot_V_node_ang,
            'S21_dB':           plot_s21_db,
            'S21_Phase_deg':    plot_s21_ang,
            'S11_dB':           plot_s11_db,
            'S11_Phase_deg':    plot_s11_ang,
        }
        self.last_results_df = pd.DataFrame(data)
        self.status_var.set("Analysis Done.")

    # --------------------------------------------------------------------------------
    # STEADY STATE ANALYSIS
    # --------------------------------------------------------------------------------
    def _run_steady_state_analysis(self, v_rms_mv, R_gen_int, R_div1, R_div2, C_line, Z_in_meas, Rser, Rload, Cload, Lload, res_params, mags, phases, n_harm):
        f0 = self._get_f_fundamental() * 1e6
        w0 = 2 * np.pi * f0
        T = 5 / f0
        t = np.linspace(0, T, 2**14, endpoint=False)
        dt = t[1] - t[0]

        # --- NEW SCALING LOGIC FOR 0V TO 1V STEADY STATES ---
        # 1. Generate the raw, unscaled waveform to analyze its actual time-domain shape
        V_raw_t = np.full_like(t, mags[0])
        for i in range(n_harm):
            V_raw_t += mags[i+1] * np.cos((i+1)*w0*t + np.deg2rad(phases[i]))

        # 2. Use percentiles to find the "flat" steady states, ignoring Gibbs ringing spikes
        V_low_raw = np.percentile(V_raw_t, 2)   # Bottom 2% represents the low steady state
        V_high_raw = np.percentile(V_raw_t, 98) # Top 98% represents the high steady state
        V_swing_raw = V_high_raw - V_low_raw

        # 3. Define target levels (0V and 1V)
        target_low = 0.0
        target_high = 1.0
        target_swing = target_high - target_low

        # 4. Calculate exact scale and DC shift needed
        scale_factor = target_swing / V_swing_raw if V_swing_raw > 1e-12 else 1.0
        
        mags_scaled = list(mags)
        for i in range(1, len(mags_scaled)):
            mags_scaled[i] *= scale_factor
            
        # Shift the DC component (mags[0]) so the low state sits exactly at target_low
        mags_scaled[0] = (mags[0] - V_low_raw) * scale_factor + target_low
        mags = mags_scaled
        
        # 5. Build the final V_source_t with our newly calibrated magnitudes
        V_source_t = np.full_like(t, mags[0])
        for i in range(n_harm):
            V_source_t += mags[i+1] * np.cos((i+1)*w0*t + np.deg2rad(phases[i]))
        # ----------------------------------------------------
            
        V_out_t = np.zeros_like(t)
        V_node_t = np.zeros_like(t)
        V_node_ref_t = np.zeros_like(t) 
        V_out_ref_t = np.zeros_like(t) 
        Iser_t = np.zeros_like(t)
        Iref_t = np.zeros_like(t)
        I_res_t_list = [np.zeros_like(t) for _ in res_params] 
        
        for n in range(n_harm + 1):
            f_h = n * f0; w_h = 2 * np.pi * f_h
            if n == 0: V_src_ph = mags[0]
            else: V_src_ph = mags[n] * np.exp(1j * np.deg2rad(phases[n-1]))
            
            if np.abs(V_src_ph) < 1e-12: continue

            if n == 0: 
                Y_div2 = 1.0/R_div2
            else:
                Y_div2 = 1.0/R_div2 + 1j*w_h*C_line
            
            Y_Zin = 1.0/Z_in_meas
            Y_shunt_total = Y_div2 + Y_Zin
            Z_shunt_total = 1.0/Y_shunt_total

            if n == 0:
                Z_node_equiv = 1.0/Y_shunt_total
                Z_total_DC = R_gen_int + R_div1 + Z_node_equiv
                
                I_total_DC = V_src_ph / Z_total_DC
                V_node_A_ph = V_src_ph * (Z_node_equiv / (R_gen_int + R_div1 + Z_node_equiv)) 
                
                Iser_phasor_h = 0j 
                Vout_phasor_h = V_node_A_ph 
                Iref_phasor_h = 0j 
                V_node_A_ref = V_node_A_ph
                Vout_phasor_h_ref = V_node_A_ph
                I_res_phasors = [0j] * len(res_params)
            else:
                Z_load_h = Rload + 1j*w_h*Lload + 1/(1j*w_h*Cload)
                Y_res = sum(1/self.calculate_single_resonator_impedance(f_h, p) for p in res_params)
                Z_res_total = 1.0/Y_res if Y_res!=0 else np.inf
                
                if Z_res_total == np.inf: Z_struct = Z_load_h
                else: Z_struct = 1.0 / (1.0/Z_load_h + 1.0/Z_res_total)
                
                Z_branch = Rser + Z_struct
                
                Z_node_equiv = (Z_shunt_total * Z_branch) / (Z_shunt_total + Z_branch)
                Z_divider_in = R_div1 + Z_node_equiv
                Z_total = R_gen_int + Z_divider_in
                
                I_total = V_src_ph / Z_total
                V_input_divider = V_src_ph - (I_total * R_gen_int)
                V_node_A_ph = V_input_divider * (Z_node_equiv / (R_div1 + Z_node_equiv))
                
                Iser_phasor_h = V_node_A_ph / Z_branch
                Vout_phasor_h = Iser_phasor_h * Z_struct
                
                I_res_phasors = [Vout_phasor_h / self.calculate_single_resonator_impedance(f_h, p) for p in res_params]

                Z_branch_ref = Rser + Z_load_h
                Z_node_equiv_ref = (Z_shunt_total * Z_branch_ref) / (Z_shunt_total + Z_branch_ref)
                Z_total_ref = R_gen_int + R_div1 + Z_node_equiv_ref
                
                I_total_ref = V_src_ph / Z_total_ref
                V_input_div_ref = V_src_ph - (I_total_ref * R_gen_int)
                V_node_A_ref = V_input_div_ref * (Z_node_equiv_ref / (R_div1 + Z_node_equiv_ref))
                Iref_phasor_h = V_node_A_ref / Z_branch_ref
                Vout_phasor_h_ref = Iref_phasor_h * Z_load_h

            V_node_t += self.to_time_domain(V_node_A_ph, w_h, t)
            V_out_t += self.to_time_domain(Vout_phasor_h, w_h, t)
            V_node_ref_t += self.to_time_domain(V_node_A_ref, w_h, t)
            V_out_ref_t += self.to_time_domain(Vout_phasor_h_ref, w_h, t)
            
            Iser_t += self.to_time_domain(Iser_phasor_h, w_h, t)
            Iref_t += self.to_time_domain(Iref_phasor_h, w_h, t)
            for i in range(len(res_params)):
                I_res_t_list[i] += self.to_time_domain(I_res_phasors[i], w_h, t)

        E_res = np.cumsum(Iser_t**2 * Rser) * dt
        E_ref = np.cumsum(Iref_t**2 * Rser) * dt
        FoM = (E_ref[-1] - E_res[-1])/E_ref[-1]*100 if E_ref[-1]>1e-20 else 0

        P_ser_t = (Iser_t**2) * Rser
        P_ref_t = (Iref_t**2) * Rser

        # Apply Subtraction if needed
        if self.subtract_ref_var.get():
            plot_V_node = V_node_t - V_node_ref_t
            plot_V_out = V_out_t - V_out_ref_t
            plot_Iser = Iser_t - Iref_t
            plot_P_ser = P_ser_t - P_ref_t
            y_mod = " (Delta)"
        else:
            plot_V_node = V_node_t
            plot_V_out = V_out_t
            plot_Iser = Iser_t
            plot_P_ser = P_ser_t
            y_mod = ""

        def _autoscale_ax(ax):
            """Force tight autoscale ignoring any stale layout constraints."""
            ax.relim()
            ax.autoscale_view(tight=True)
            # Add 10% padding on Y
            y0, y1 = ax.get_ylim()
            pad = (y1 - y0) * 0.10 if y1 != y0 else 1.0
            ax.set_ylim(y0 - pad, y1 + pad)

        if self.show_vgen_var.get():
            self.axes[0].plot(t*1e9, V_source_t, label='V_gen', color='green', alpha=0.5)
        self.axes[0].plot(t*1e9, plot_V_node, label='V_node' + y_mod, color='orange')
        self.axes[0].plot(t*1e9, plot_V_out, label='V_out' + y_mod, linestyle='--', color='blue')
        self.axes[0].set_title(f'Voltage Overlay{y_mod}'); self.axes[0].set_ylabel('V'); self.axes[0].legend(fontsize=7, loc='best'); self.axes[0].grid(True)
        _autoscale_ax(self.axes[0])

        self.axes[1].plot(t*1e9, plot_Iser*1000, label='I_ser' + y_mod, color='black', linewidth=2)
        if not self.subtract_ref_var.get():
             self.axes[1].plot(t*1e9, Iref_t*1000, label='I_ref', linestyle=':', color='gray')
        for i, i_res in enumerate(I_res_t_list):
             self.axes[1].plot(t*1e9, i_res*1000, label=f'I_res{i+1}', linestyle='--')
        self.axes[1].set_title(f'Currents{y_mod}'); self.axes[1].set_ylabel('mA'); self.axes[1].legend(fontsize=7, loc='best'); self.axes[1].grid(True)
        _autoscale_ax(self.axes[1])

        self.axes[2].plot(t*1e9, E_ref*1e12, label='Ref (No Res)')
        self.axes[2].plot(t*1e9, E_res*1e12, label='With Res')
        self.axes[2].set_title(f'Energy (FoM={FoM:.2f}%)'); self.axes[2].set_ylabel('pJ'); self.axes[2].legend(fontsize=7, loc='best'); self.axes[2].grid(True)
        self.axes[2].set_xlabel('')
        _autoscale_ax(self.axes[2])

        if not self.subtract_ref_var.get():
            self.axes[3].plot(t*1e9, P_ref_t*1000, label='P_ref', color='gray', linestyle=':')
        self.axes[3].plot(t*1e9, plot_P_ser*1000, label='P_ser' + y_mod, color='red')
        self.axes[3].set_title(f'Power Dissipation{y_mod}'); self.axes[3].set_ylabel('mW')
        self.axes[3].legend(fontsize=7, loc='best'); self.axes[3].grid(True); self.axes[3].set_xlabel('Time (ns)')
        _autoscale_ax(self.axes[3])
        
        self.last_filename = self._build_save_filename(
            'steady', f0, res_params, fom=FoM)
        
        data = {
            'Time_s': t, 'V_source': V_source_t, 'V_node': V_node_t, 'V_out': V_out_t, 
            'I_ser': Iser_t, 'P_ser_W': P_ser_t, 'P_ref_W': P_ref_t
        }
        for i, i_res in enumerate(I_res_t_list): data[f'I_res{i+1}'] = i_res
        
        self.last_results_df = pd.DataFrame(data)
        self.status_var.set("Steady State Analysis Done.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ResonatorApp(root)
    root.mainloop()