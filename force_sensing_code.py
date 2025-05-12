import marimo

__generated_with = "0.11.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# **Force Sensing**""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### **Overview**""")
    return


@app.cell(hide_code=True)
def _(mo, os):
    _src = (os.getcwd()) + '\\system_schematic.png'
    mo.image(src=_src, width="500px", height="309px", rounded=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### **Problem Setup**""")
    return


@app.cell(hide_code=True)
def _():
    # IMPORTS
    import marimo as mo
    import os
    from math import pi, sqrt
    import ast
    from PIL import Image
    from pyslsqp import optimize
    import plotly.graph_objects as go
    from scipy.optimize import minimize
    from scipy.optimize import differential_evolution
    from scipy.interpolate import interp1d
    import serial
    import time
    import matlab.engine
    import numpy as np
    from smt.surrogate_models import KRG, RBF, KPLSK
    from smt.sampling_methods import LHS
    from sklearn.metrics import mean_squared_error
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm, Normalize
    import asyncio
    eng = matlab.engine.start_matlab()
    return (
        Image,
        KPLSK,
        KRG,
        LHS,
        Normalize,
        RBF,
        StandardScaler,
        TwoSlopeNorm,
        ast,
        asyncio,
        differential_evolution,
        eng,
        go,
        interp1d,
        matlab,
        mean_squared_error,
        minimize,
        mo,
        np,
        optimize,
        os,
        pi,
        plt,
        serial,
        sqrt,
        time,
    )


@app.cell(hide_code=True)
def _(mo):
    # GEOMETRY PRESET INPUT
    dropdown_geometry_preset_input = mo.ui.dropdown(options=["Optimized Beam", "Pre-Curved Beam", "Straight Constant R Beam", "Tapered Beam", "Custom Beam"], label="choose geometry preset")
    return (dropdown_geometry_preset_input,)


@app.cell(hide_code=True)
def _(dropdown_geometry_preset_input, mo, n, np):
    # GEOMETRY PRESET ARRAYS
    array_dict = {"x": "[0]", "y": "[0]", "z": "[0]", "moment": "[0]"}
    if dropdown_geometry_preset_input.value == "Optimized Beam":
        array_dict["x"] = str(np.linspace(0, 0.2, n).tolist())
        array_dict["y"] = str(np.zeros(n).tolist())
        array_dict["z"] = str(np.zeros(n).tolist())
        array_dict["moment"] = "[1.588, 0.9782, 1.546, 3.305, 12.00, 6.289, 4.179, 8.636, 3.424, 6.742,  12.10, 4.875, 8.867, 6.178, 10.08, 3.867, 4.634, 11.37, 1.688, 6.272]"
    elif dropdown_geometry_preset_input.value == "Pre-Curved Beam":
        x_start = 0
        x_end = 188
        x_arr = np.linspace(x_start, x_end, n)
        y_arr = np.zeros_like(x_arr)
        for i, x in enumerate(x_arr):
            if x <= 10:
                y_arr[i] = 0  # flat part
            else:
                y_arr[i] = -0.001894*x**2 + 0.03788*x - 0.1894  # parabolic part
        x_arr, y_arr = x_arr/1000, y_arr/1000 # convert to m

        array_dict["x"] = np.array2string(x_arr, separator=', ')
        array_dict["y"] = np.array2string(y_arr, separator=', ')

        array_dict["z"] = str(np.zeros(n).tolist())
        array_dict["moment"] = str(np.full(20, 1.258).tolist()) # 1.125mm R constant
    elif dropdown_geometry_preset_input.value == "Straight Constant R Beam":
        array_dict["x"] = str(np.linspace(0, 0.2, n).tolist())
        array_dict["y"] = str(np.zeros(n).tolist())
        array_dict["z"] = str(np.zeros(n).tolist())
        array_dict["moment"] = str(np.full(20, 1.258).tolist()) # 1.125mm R constant
    elif dropdown_geometry_preset_input.value == "Tapered Beam":
        array_dict["x"] = "[3]"
        array_dict["y"] = "[0]"
        array_dict["z"] = "[0]"
        array_dict["moment"] = "[0]"
    elif dropdown_geometry_preset_input.value == "Custom Beam":
        array_dict["x"] = "[4]"
        array_dict["y"] = "[0]"
        array_dict["z"] = "[0]"
        array_dict["moment"] = "[0]"
    else:
        array_dict["x"] = "[0]"
        array_dict["y"] = "[0]"
        array_dict["z"] = "[0]"
        array_dict["moment"] = "[0]"

    x_array_input = mo.ui.text(value=array_dict["x"], placeholder=array_dict["x"], label=r"x-coordinate array, $x_{array}$ [m]")
    y_array_input = mo.ui.text(value=array_dict["y"], placeholder=array_dict["y"], label=r"y-coordinate array, $y_{array}$ [m]")
    z_array_input = mo.ui.text(value=array_dict["z"], placeholder=array_dict["z"], label=r"z-coordinate array, $z_{array}$ [m]")
    moment_array_input = mo.ui.text(value=array_dict["moment"], placeholder=array_dict["moment"], label=r"moment array, $I_{array}$ [mm^4]")
    return (
        array_dict,
        i,
        moment_array_input,
        x,
        x_arr,
        x_array_input,
        x_end,
        x_start,
        y_arr,
        y_array_input,
        z_array_input,
    )


@app.cell(hide_code=True)
def _(
    ast,
    calculate_button,
    matlab,
    moment_array_input,
    n,
    np,
    x_array_input,
    y_array_input,
    z_array_input,
):
    # GEOMETRY VARIABLE ASSIGNMENT
    x_array = np.array(ast.literal_eval(x_array_input.value))
    y_array = -1*np.array(ast.literal_eval(y_array_input.value))
    z_array = np.array(ast.literal_eval(z_array_input.value))
    moment_array = np.array(ast.literal_eval(moment_array_input.value))
    # calculate total beam length
    points = np.vstack((x_array, y_array, z_array)).T # Stack into a 3D point array of shape (20, 3)
    diffs = np.diff(points, axis=0) # Compute the difference between consecutive points
    segment_lengths = np.linalg.norm(diffs, axis=1) # Compute the Euclidean distances between each pair of points
    L = np.sum(segment_lengths) # Sum all segment lengths to get the total beam length

    if calculate_button.value == 1:
        if not(len(x_array) == len(y_array) == len(z_array) == n):
            print("Error: geometry arrays dimension must match number of nodes!")
            raise
        else:
            x_discretized =  matlab.double(x_array.tolist())
            y_discretized =  matlab.double(y_array.tolist())
            z_discretized =  matlab.double(z_array.tolist())
        # if using less moment array elements than nodes   
        if len(moment_array) == n:
            # converts user input mm^4 to m^4 which Elastica takes
            I_discretized = 1e-12*moment_array
            J_discretized = 2*I_discretized
        elif len(moment_array) < n:
            # repeats each element so that it fits #of nodes
            repeat_factor = int(n/len(moment_array))
            # converts user input mm^4 to m^4 which Elastica takes
            I_discretized = np.repeat(1e-12*moment_array, repeat_factor) 
            J_discretized = 2*I_discretized # THIS RELATIONSHIP ONLY HOLDS FOR CIRCULAR CROSS SECTION!
        elif len(moment_array) > n:
            print("Error: cannot have more moment array elemets than nodes!")
            raise
    return (
        I_discretized,
        J_discretized,
        L,
        diffs,
        moment_array,
        points,
        repeat_factor,
        segment_lengths,
        x_array,
        x_discretized,
        y_array,
        y_discretized,
        z_array,
        z_discretized,
    )


@app.cell(hide_code=True)
def _(
    dropdown_geometry_preset_input,
    mo,
    moment_array_input,
    x_array_input,
    y_array_input,
    z_array_input,
):
    # DISPLAY GEOMETRY
    geometry_stack = mo.vstack([
        dropdown_geometry_preset_input,
        x_array_input,
        y_array_input,
        z_array_input,
        moment_array_input
    ])

    mo.accordion({r"**Geometry**": geometry_stack})
    return (geometry_stack,)


@app.cell(hide_code=True)
def _(mo):
    # PARAMETER INPUTS
    n_input = mo.ui.text(value="60", placeholder="60", label=r"Number of points")
    g_input = mo.ui.text(value="9.80665", placeholder="9.80665", label=r"Gravitational acceleration, $g$ [m/s²]")
    E_input = mo.ui.text(value="2.18e9", placeholder="2.18e9", label=r"Young's Modulus, $E$ [Pa]")
    G_input = mo.ui.text(value="2.4e9", placeholder="2.4e9", label=r"Shear modulus, $G$ [Pa]")
    return E_input, G_input, g_input, n_input


@app.cell(hide_code=True)
def _(E_input, G_input, g_input, n_input):
    # PARAMETER VARIABLE ASSIGNMENT
    n = int(n_input.value)
    g = float(g_input.value)
    E = float(E_input.value)
    G = float(G_input.value)
    return E, G, g, n


@app.cell(hide_code=True)
def _(E_input, G_input, g_input, mo, n_input):
    # DISPLAY PARAMETERS
    parameters_stack = mo.vstack([
    mo.hstack([n_input, g_input]),
    mo.hstack([G_input, E_input]),
     ])

    mo.accordion({r"**Parameters**": parameters_stack})
    return (parameters_stack,)


@app.cell(hide_code=True)
def _(mo, pi):
    # EXPERIMENT INPUTS
    dropdown_sensor = mo.ui.dropdown(options=["Load Cell Sensing", "Load Cell Validation", "Single Simulation", "Robust Simulation", "Robust Optimization", "Surrogate Model Sampling"], label="choose experiment type")
    F_applied_input = mo.ui.text(value=str(0.2), placeholder=str(0.2), label=r"Force applied, $F$ [N]")
    s_applied_input = mo.ui.text(value= str(0.2), placeholder=str(0.2), label=r"Arc-length distance, $s$ [m]")
    z_applied_input = mo.ui.text(value=str(pi), placeholder=str(pi), label=r"Angle zeta, $\zeta$ [rad]")
    calculate_button = mo.ui.run_button(label="Calculate")
    F_applied_lower_input = mo.ui.text(value=str(0.1), placeholder=str(0.1), label=r"Force applied (lower bound), $F$ [N]")
    F_applied_upper_input = mo.ui.text(value=str(0.3), placeholder=str(0.3), label=r"Force applied (upper bound), $F$ [N]")
    s_applied_lower_input = mo.ui.text(value=str(0.075), placeholder=str(0.075), label=r"Arc-length distance (lower bound), $s$ [m]")
    s_applied_upper_input = mo.ui.text(value=str(0.2), placeholder=str(0.2), label=r"Arc-length distance (upper bound), $s$ [m]")
    num_input_samples_input = mo.ui.text(value="100", placeholder="100", label=r"Number of Force-Position Samples")
    num_design_vars_input = mo.ui.text(value="20", placeholder="20", label=r"Number of Design Variables")
    I_min_input = mo.ui.text(value="0.032", placeholder="0.032", label=r"I_min $I$ [mm^4]")
    I_max_input = mo.ui.text(value="0.32", placeholder="0.32", label=r"I_max $I$ [mm^4]")
    R_min_input = mo.ui.text(value="0.25", placeholder="0.25", label=r"R_min $R$ [mm]")
    R_max_input = mo.ui.text(value="3", placeholder="3", label=r"R_max $R$ [mm]")
    Fx_error_mean_input = mo.ui.text(value="0", placeholder="0", label=r"$E[Fx_{\text{error}}]$ [N]")
    Fx_error_stdev_input = mo.ui.text(value="0.0025", placeholder="0.0025", label=r"$σ[Fx_{\text{error}}]$ [N]")
    MB_error_mean_input = mo.ui.text(value="0", placeholder="0", label=r"$E[MB_{\text{error}}]$ [Nm]")
    MB_error_stdev_input = mo.ui.text(value="0.0005", placeholder="0.0005", label=r"$σ[MB_{\text{error}}]$ [Nm]")
    num_uncertainty_samples_input =  mo.ui.text(value=str(5), placeholder=str(5), label=r"Number of random samples for uncertainty in Fx and MB")
    robust_constraint_input = mo.ui.dropdown(options=["second moment of area", "radius"], label="choose constraint type")
    calculation_method_input = mo.ui.dropdown(options=["explicit", "surrogate model"], label="choose method for calculating objective function")
    return (
        F_applied_input,
        F_applied_lower_input,
        F_applied_upper_input,
        Fx_error_mean_input,
        Fx_error_stdev_input,
        I_max_input,
        I_min_input,
        MB_error_mean_input,
        MB_error_stdev_input,
        R_max_input,
        R_min_input,
        calculate_button,
        calculation_method_input,
        dropdown_sensor,
        num_design_vars_input,
        num_input_samples_input,
        num_uncertainty_samples_input,
        robust_constraint_input,
        s_applied_input,
        s_applied_lower_input,
        s_applied_upper_input,
        z_applied_input,
    )


@app.cell(hide_code=True)
def _(
    F_applied_input,
    F_applied_lower_input,
    F_applied_upper_input,
    Fx_error_mean_input,
    Fx_error_stdev_input,
    I_max_input,
    I_min_input,
    MB_error_mean_input,
    MB_error_stdev_input,
    R_max_input,
    R_min_input,
    ast,
    calculation_method_input,
    mo,
    np,
    num_design_vars_input,
    num_input_samples_input,
    num_uncertainty_samples_input,
    robust_constraint_input,
    s_applied_input,
    s_applied_lower_input,
    s_applied_upper_input,
    z_applied_input,
):
    # EXPERIMENT VARIABLE ASSIGNMENT
    F_applied = float(F_applied_input.value)
    s_applied = float(s_applied_input.value)
    z_applied = float(z_applied_input.value)
    F_applied_lower = float(F_applied_lower_input.value)
    F_applied_upper = float(F_applied_upper_input.value)
    s_applied_lower = float(s_applied_lower_input.value)
    s_applied_upper = float(s_applied_upper_input.value)
    num_input_samples = int(num_input_samples_input.value)
    num_design_vars =  int(num_design_vars_input.value)
    R_min = float(R_min_input.value)#[mm]
    R_max = float(R_max_input.value)#[mm]
    Fx_error_mean = float(Fx_error_mean_input.value)
    Fx_error_stdev = float(Fx_error_stdev_input.value)
    MB_error_mean = float(MB_error_mean_input.value)
    MB_error_stdev = float(MB_error_stdev_input.value)
    num_uncertainty_samples = int(num_uncertainty_samples_input.value)

    calculation_method = str(calculation_method_input.value)

    if robust_constraint_input.value == "second moment of area":
        constraint_stack = mo.hstack([I_min_input, I_max_input])
        I_min = float(ast.literal_eval(I_min_input.value)) #[mm^4]
        I_max = float(ast.literal_eval(I_max_input.value)) #[mm^4]
    elif robust_constraint_input.value == "radius":
        constraint_stack = mo.hstack([R_min_input, R_max_input])
        I_min = (np.pi*R_min**4)/4 #[mm^4]
        I_max = (np.pi*R_max**4)/4 #[mm^4]
    else:
        constraint_stack = mo.hstack([])
    return (
        F_applied,
        F_applied_lower,
        F_applied_upper,
        Fx_error_mean,
        Fx_error_stdev,
        I_max,
        I_min,
        MB_error_mean,
        MB_error_stdev,
        R_max,
        R_min,
        calculation_method,
        constraint_stack,
        num_design_vars,
        num_input_samples,
        num_uncertainty_samples,
        s_applied,
        s_applied_lower,
        s_applied_upper,
        z_applied,
    )


@app.cell(hide_code=True)
def _(
    F_applied_input,
    F_applied_lower_input,
    F_applied_upper_input,
    Fx_error_mean_input,
    Fx_error_stdev_input,
    MB_error_mean_input,
    MB_error_stdev_input,
    calculate_button,
    calculation_method_input,
    constraint_stack,
    dropdown_sensor,
    mo,
    n,
    num_design_vars,
    num_design_vars_input,
    num_input_samples_input,
    num_uncertainty_samples_input,
    robust_constraint_input,
    s_applied_input,
    s_applied_lower_input,
    s_applied_upper_input,
    z_applied_input,
):
    # DISPLAY EXPERIMENT
    sensing_mode = None
    if dropdown_sensor.value == "Load Cell Sensing":
        sensor_stack = mo.hstack([calculate_button])
        calculate_button
    elif dropdown_sensor.value == "Load Cell Validation":
        sensor_stack = mo.vstack([
            mo.hstack([F_applied_input, s_applied_input, z_applied_input]),
            calculate_button
        ])
    elif dropdown_sensor.value == "Single Simulation":
        sensor_stack = mo.vstack([
            mo.hstack([F_applied_input, s_applied_input, z_applied_input]),
            calculate_button
        ])
    elif dropdown_sensor.value == "Robust Simulation": 
       # if design variables divides into nodes with integer result
        if num_design_vars < n and n % num_design_vars == 0:
            nodes_per_design_vars = n // num_design_vars    
        sensor_stack = mo.vstack([
            z_applied_input,
            mo.hstack([F_applied_lower_input, F_applied_upper_input]),
            mo.hstack([s_applied_lower_input, s_applied_upper_input]),
            mo.hstack([Fx_error_mean_input, Fx_error_stdev_input]),
            mo.hstack([MB_error_mean_input, MB_error_stdev_input]),
            num_uncertainty_samples_input,
            num_input_samples_input,
            calculation_method_input,
            calculate_button
        ])
    elif dropdown_sensor.value == "Robust Optimization" or dropdown_sensor.value == "Surrogate Model Sampling":
        # if design variables divides into nodes with integer result
        if num_design_vars < n and n % num_design_vars == 0:
            nodes_per_design_vars = n // num_design_vars    
        sensor_stack = mo.vstack([
            z_applied_input,
            mo.hstack([F_applied_lower_input, F_applied_upper_input]),
            mo.hstack([s_applied_lower_input, s_applied_upper_input]),
            mo.hstack([Fx_error_mean_input, Fx_error_stdev_input]),
            mo.hstack([MB_error_mean_input, MB_error_stdev_input]),
            mo.hstack([num_input_samples_input, num_design_vars_input]),
            num_uncertainty_samples_input,
            robust_constraint_input,
            constraint_stack,
            calculation_method_input,
            calculate_button
        ])
    else:
        sensor_stack = mo.hstack([])

    all_stack = mo.vstack([dropdown_sensor, sensor_stack]) 
    mo.accordion({r"**Experiment Setup**": all_stack})
    return all_stack, nodes_per_design_vars, sensing_mode, sensor_stack


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### **Force Sensing Results**""")
    return


@app.cell(hide_code=True)
def _(
    E,
    F_applied,
    G,
    I_discretized,
    J_discretized,
    calculate_button,
    dropdown_sensor,
    eng,
    get_loadcell_values,
    mo,
    n,
    os,
    s_applied,
    x_discretized,
    y_discretized,
    z_applied,
    z_discretized,
):
    # RUN EXPERIMENT ANS DISPLAY RESULTS
    Fx_sim = MB_sim = MD_sim = None
    Fx_cell = MB_cell = MD_cell = None
    F_estimate_sim = s_estimate_sim = z_estimate_sim = None
    F_estimate_cell = s_estimate_cell = z_estimate_cell = None
    plot1 = plot2 = plot3 = None

    # if calculate button is pressed
    if calculate_button.value == 1:
        if dropdown_sensor.value == "Load Cell Sensing":
            # Load Cell data retrieval via serial communication
            [Fx_cell, MB_cell, MD_cell] = get_loadcell_values()
            # Estimate applied force from known proximal-end load cell values
            [_, _, _, F_estimate_cell, s_estimate_cell, z_estimate_cell] = eng.force_estimation_from_cell(n, x_discretized, y_discretized, z_discretized, E, I_discretized, G, J_discretized, Fx_cell, MB_cell, MD_cell, True, nargout = 6)
        elif dropdown_sensor.value == "Load Cell Validation":
            # Load Cell data retrieval via serial communication
            [Fx_cell, MB_cell, MD_cell] = get_loadcell_values()
            Fx_cell = -0.016
            MB_cell = 0.008
            # Estimate applied force from known proximal-end load cell values
            [_, _, _, F_estimate_cell, s_estimate_cell, z_estimate_cell] = eng.force_estimation_from_cell(n, x_discretized, y_discretized, z_discretized, E, I_discretized, G, J_discretized, Fx_cell, MB_cell, MD_cell, True, nargout = 6)
            # Estimate applied force using simulated load
            [Fx_sim, MB_sim, MD_sim, _, _, _, F_estimate_sim, s_estimate_sim, z_estimate_sim] = eng.force_estimation_from_simulation(n, x_discretized, y_discretized, z_discretized, F_applied, s_applied, z_applied, E, I_discretized, G, J_discretized, True, 0, 0, nargout = 9)
        elif dropdown_sensor.value == "Single Simulation":
            # Estimate applied force using simulated load
            [Fx_sim, MB_sim, MD_sim, _, _, _, F_estimate_sim, s_estimate_sim, z_estimate_sim] = eng.force_estimation_from_simulation(n, x_discretized, y_discretized, z_discretized, F_applied, s_applied, z_applied, E, I_discretized, G, J_discretized, True, 0, 0, nargout = 9)
        elif dropdown_sensor.value == "Robust Simulation":
            pass

        # Load MATLAB plots
        _src1 = (os.getcwd()) + '\\plot1.png'
        plot1 = mo.image(src=_src1, width="500px", height="500px", rounded=True)
        _src2 = (os.getcwd()) + '\\plot2.png'
        plot2 = mo.image(src=_src2, width="500px", height="500px", rounded=True)
        _src3 = (os.getcwd()) + '\\plot3.png'
        plot3 = mo.image(src=_src3, width="500px", height="500px", rounded=True)

    sensor_table = mo.ui.table(
        data=[
            {"Name": "Fx cell", "Value": Fx_cell},
            {"Name": "Fx simulation", "Value": Fx_sim},
            {"Name": "MB cell", "Value": MB_cell},
            {"Name": "MB simulation", "Value": MB_sim},
            {"Name": "MD cell", "Value": MD_cell},
            {"Name": "MD simulation", "Value": MD_sim},
        ], page_size = 9, label="Sensor Measurements")

    force_table = mo.ui.table(
        data=[
            {"Name": "F estimate from cell", "Value": F_estimate_cell},
            {"Name": "F estimate from simulation", "Value": F_estimate_sim},
            {"Name": "s estimate from cell", "Value": s_estimate_cell},
            {"Name": "s estimate from simulation", "Value": s_estimate_sim},
        ], page_size = 9, label="Force Measurements")

    plot_stack = mo.vstack([
        mo.hstack([plot1, plot2]),
        mo.hstack([plot3]),
        mo.hstack([force_table, sensor_table])
    ])

    mo.accordion({r"**Results**": plot_stack})
    return (
        F_estimate_cell,
        F_estimate_sim,
        Fx_cell,
        Fx_sim,
        MB_cell,
        MB_sim,
        MD_cell,
        MD_sim,
        force_table,
        plot1,
        plot2,
        plot3,
        plot_stack,
        s_estimate_cell,
        s_estimate_sim,
        sensor_table,
        z_estimate_cell,
        z_estimate_sim,
    )


@app.cell(hide_code=True)
def _(calculate_button, np, serial, time):
    def get_loadcell_values():
        # Load Cell data retrieval via serial communication
        if calculate_button.value == 1:
            print("connecting...")
            # Configure the arduino serial connection
            arduino_port = "COM3"
            arduino_baud_rate = 57600       # Match the baud rate set in the Arduino code
            arduino_timeout = 2            # Timeout in seconds for the serial connection
            # Initialize the serial connection
            arduino_serial = serial.Serial(port=arduino_port, baudrate=arduino_baud_rate, timeout=arduino_timeout)
            # Allow some time for the connection to establish
            time.sleep(2)
            # Configure the MARK-10 serial connection
            mark10_port = "COM4"
            mark10_baud_rate = 115200       # Match the baud rate set in the Arduino code
            mark10_timeout = 2            # Timeout in seconds for the serial connection
            # Initialize the serial connection
            mark10_serial = serial.Serial(port=mark10_port, baudrate=mark10_baud_rate, timeout=mark10_timeout)
            # Allow some time for the connection to establish
            time.sleep(2)
            # Initialize stability tracking
            num_stable_loops = 20  # Number of consecutive stable readings
            threshold = 0.001  # Minimum value for absolute stability check
            fluctuation_limit_percentage = 0.05  # Allowable fluctuation as a percentage
            stable_count = 0
            previous_values = [0, 0, 0]  # To store previous stable values of x, y, z
            Fx_total, Fy_total, Fz_total = 0, 0, 0  # Accumulators for averaging
            first_time = True
            arduino_finished = False # false by default until measuments have been made
            mark10_finished = False # false by default until measuments have been made
            while not arduino_finished or not mark10_finished:
                if arduino_serial.in_waiting > 0 and not arduino_finished:
                    if first_time:
                        print("Please add a load...")
                        first_time = False
                    line = arduino_serial.readline().decode('utf-8').strip()  # Read and decode the line
                    try:
                        # Step 1: get x_cell value
                        x_cell = float(line) # force in [N]
                        # Check if any of the values exceed the threshold
                        if any(abs(val) > threshold for val in [x_cell]):
                            # Calculate fluctuation relative to the previous stable values
                            fluctuations = [
                                abs(val - prev)
                                for val, prev in zip([x_cell], previous_values)
                            ]
                            # Check if all fluctuations are within the allowable range
                            if all(fluct <= abs(fluctuation_limit_percentage*value) for fluct, value in zip(fluctuations, [x_cell])):
                                print(f"x: {x_cell}")
                                # Accumulate stable values
                                Fx_total += x_cell
                                stable_count += 1
                                if stable_count == num_stable_loops and mark10_finished:
                                    arduino_finished = True
                            else:
                                stable_count = 0  # Reset if fluctuation exceeds limit
                                previous_values = [x_cell]
                                Fx_total = 0  # Reset accumulators
                        else:
                            stable_count = 0  # Reset if values drop below the threshold
                            Fx_total = 0  # Reset accumulators
                    except ValueError:
                        print(f"Invalid data received: {line}")
                if mark10_serial.in_waiting > 0 and not mark10_finished:
                    line = mark10_serial.readline()
                    try:
                        My = float(line)*0.01 # convert [Ncm] to [Nm]
                        print(f"M_y: {My}")
                        stable_count = 0 
                        Fx_total = 0
                        mark10_finished = True
                    except ValueError:
                        print(f"Invalid data received: {line}")

            print(f"Success! Stable readings achieved for {num_stable_loops:.1f} loops")
            # Calculate averages
            Fx_avg = Fx_total / num_stable_loops

            Fx_cell = Fx_avg;
            My_cell = My # for now (change later) 
            Mz_cell = 0 # for now (change later)
            MB_cell = np.linalg.norm(np.array([My_cell,Mz_cell]))
            MD_cell = np.arctan2(My_cell, Mz_cell);
            return [Fx_cell, MB_cell, MD_cell]
        else:
            # if calculate_button has not been pressend yet
            return [None, None, None]
    return (get_loadcell_values,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### **Optimization Results**""")
    return


@app.cell(hide_code=True)
def _(
    E,
    F_applied_lower,
    F_applied_upper,
    Fx_error_mean,
    Fx_error_stdev,
    G,
    I_max,
    I_min,
    KPLSK,
    KRG,
    LHS,
    MB_error_mean,
    MB_error_stdev,
    Normalize,
    RBF,
    StandardScaler,
    TwoSlopeNorm,
    calculate_button,
    calculation_method,
    differential_evolution,
    dropdown_sensor,
    eng,
    go,
    mo,
    moment_array,
    n,
    nodes_per_design_vars,
    np,
    num_design_vars,
    num_input_samples,
    num_uncertainty_samples,
    plt,
    s_applied_lower,
    s_applied_upper,
    x_discretized,
    y_discretized,
    z_applied,
    z_discretized,
):
    class RobustOpt():
        def __init__(self):
            # Global Variable to Track Best Solution
            self.best_solution = {'I_array': np.full(num_design_vars, None),'objective_value': np.inf, 'delta_F_mean': None, 'delta_s_mean': None, 'delta_F_stdev': None, 'delta_s_stdev': None, 'I_plot': None, 'R_plot': None, 'R_array': np.full(num_design_vars, None), 'convergence_plot': None, 'fun_per_iteration': [], 'I_array_per_iteration': [], 'delta_F_array': [], 'delta_s_array': []}
            # Change Print Options
            np.set_printoptions(precision=3, suppress=True)
            self.iteration_counter = 1
            self.sub_iteration_counter = 1
            self.show_sub_plots = True # show best plot in console when it is found
            self.custom_iteration = True # show custom iteration info print
            self.model = None # surrogate model
            self.convergence_tol = 1e-8
            self.max_iter = 1400
            np.random.seed(42) # Ensures that random numbers are the same every time
            self.F_samples = np.full(num_input_samples, None) # None by default until calculated 
            self.s_samples = np.full(num_input_samples, None) # None by default until calculated 

        def get_data(self):
            return self.best_solution['I_plot'], self.best_solution['I_array'], self.best_solution['objective_value'], self.best_solution['R_plot'], self.best_solution['convergence_plot'], self.best_solution['R_array'], self.best_solution['delta_F_stdev'], self.best_solution['delta_s_stdev'] 

        def generate_input_samples(self):
            # Define the input space bounds
            input_limits = np.array([[F_applied_lower, F_applied_upper], [s_applied_lower, s_applied_upper]]) 
            sampling = LHS(xlimits=input_limits, criterion='maximin')
            input_samples = sampling(num_input_samples)  # Generate initial training points
            self.F_samples = np.array(input_samples[:, 0]) # get all sampeld F's
            self.s_samples = np.array(input_samples[:, 1]) # get all sampeld s's

        def load_model(self, modelObject):
            self.model = modelObject

        # Calculate Maximum Error
        def calculate_objective(self, I_robust): 
            if self.model == None: # if no model provided, explicitly calculate results
                if self.F_samples.any() == None or  self.s_samples.any() == None: self.generate_input_samples() # ensures samples exist
                delta_F_array = np.zeros(num_input_samples) # reset each iteration
                delta_s_array = np.zeros(num_input_samples) # reset each iteration
                # if using less design vars than nodes
                if num_design_vars < n:
                    # repeats each element so that it fits #of nodes
                    # converts user input mm^4 to m^4 which Elastica takes
                    I_adjusted = np.repeat(1e-12*I_robust, nodes_per_design_vars) 
                    J_adjusted = 2*I_adjusted # THIS RELATIONSHIP ONLY HOLDS FOR CIRCULAR CROSS SECTION!
                if num_design_vars == n:
                    # converts user input mm^4 to m^4 which Elastica takes
                    I_adjusted = 1e-12*I_robust
                    J_adjusted = 2*I_adjusted
                # if there is Fx (axial force) uncertainty 
                if Fx_error_stdev != 0:
                    Fx_errors_list = np.zeros((num_uncertainty_samples, num_input_samples))
                    for i in range(num_uncertainty_samples):
                        # pick a random sample from Fx uncertianty distribution for each (F,s) sample
                        Fx_errors_list[i] = np.random.normal(Fx_error_mean, Fx_error_stdev, num_input_samples)
                # if there is MB (bending moment) uncertainty
                if MB_error_stdev != 0:
                    MB_errors_list = np.zeros((num_uncertainty_samples, num_input_samples))
                    for i in range(num_uncertainty_samples):
                        # pick a random sample from Fx uncertianty distribution for each (F,s) sample
                        MB_errors_list[i] = np.random.normal(MB_error_mean, MB_error_stdev, num_input_samples)
                # Handle cases where only one or neither has uncertainty
                if Fx_error_stdev == 0 and MB_error_stdev != 0:
                    Fx_errors_list = np.zeros((num_uncertainty_samples, num_input_samples))
                elif MB_error_stdev == 0 and Fx_error_stdev != 0:
                    MB_errors_list = np.zeros((num_uncertainty_samples, num_input_samples))
                elif Fx_error_stdev == 0 and MB_error_stdev == 0:
                    Fx_errors_list = np.zeros((1, num_input_samples))
                    MB_errors_list = np.zeros((1, num_input_samples))

                # calculates objective function for each Fx MB uncertianty sample.
                delta_F_array = np.zeros((num_uncertainty_samples, num_input_samples))
                delta_s_array = np.zeros((num_uncertainty_samples, num_input_samples))
                for i, (Fx_errors, MB_errors) in enumerate(zip(Fx_errors_list, MB_errors_list)):
                    results = eng.NoahParfeval(n, x_discretized, y_discretized, z_discretized, z_applied, E,
                                        I_adjusted, G, J_adjusted, self.s_samples, self.F_samples, Fx_errors, MB_errors)
                    outputs = np.array(results)
                    for j, F in enumerate(self.F_samples):
                        F_estimate = outputs[6][j]
                        delta_F = F_estimate - F
                        delta_F_array[i,j] = delta_F
                    for j, s in enumerate(self.s_samples):
                        s_estimate = outputs[7][j] 
                        delta_s = s_estimate - s
                        delta_s_array[i,j] = delta_s

                # Calculate the sample mean for delta_F and delta_s
                delta_F_mean = np.mean(delta_F_array)
                delta_s_mean = np.mean(delta_s_array)
                # Calculate the sample stdev for delta_F and delta_s
                delta_F_stdev = np.std(delta_F_array, ddof=1)
                delta_s_stdev = np.std(delta_s_array, ddof=1)
                # Abs mean ensures that both overestimation and underestimation contribute to the objective function.
                objective_value = abs(delta_F_mean)+delta_F_stdev+abs(delta_s_mean)+delta_s_stdev
            elif self.model != None: # If model is provided use that instead
                delta_F_mean = delta_s_mean = delta_F_stdev = delta_s_stdev = None
                objective_value = self.model.predict(1e-12*I_robust.reshape(1, -1))
            return objective_value, delta_F_mean, delta_s_mean, delta_F_stdev, delta_s_stdev, delta_F_array, delta_s_array

        def generate_plot(self, x, y):
            data_2d = {'x': x,'y': y}
            # I Plot
            I_plot = go.Figure()
            # Add data trace
            I_plot.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name='Second Moment of Area'))
            I_plot.update_layout(
                title='I distribution',
                xaxis_title='Array Element Index',
                yaxis_title='Second Moment of Area [mm^4]',
                xaxis=dict(range=[1, num_design_vars]),
                yaxis=dict(range=[np.min(y),np.max(y)], tickformat=".2e"),
                showlegend=False,
                width=500,     
                height=400  
            )
            # R Plot
            R_plot = go.Figure()
            # Width of each segment (adjust as needed)
            dx = 0.5  
            x_segments = []
            y_segments = []
            for xi, yi in zip(x, y):
                x_segments.extend([xi - dx, xi + dx, None])  # Start, End, and None (break)
                yi = (4 * yi / np.pi) ** (1/4) # assumes circular cross section
                y_segments.extend([yi, yi, None])  # Constant height
            R_plot.add_trace(go.Scatter(
                x=x_segments, 
                y=y_segments,
                mode='lines',
                line=dict(color='green', width=3),
                fill='tozeroy',  # Fill area beneath the lines
                fillcolor='rgba(0, 255, 0, 0.3)',  # Semi-transparent blue fill
                name='Radius'
            ))
            R_plot.update_layout(
                title="Object Radius Distribution (assuming cylindrical)",
                xaxis=dict(title="Array Element Index"),
                yaxis=dict(title="Radius [mm]", tickformat=".2"),
                showlegend=False,
                width=500,
                height=400
            )
            self.best_solution['I_plot'] = I_plot # update dictionary
            self.best_solution['R_plot'] = R_plot # update dictionary

        # Objective Function: Minimize the Range of Moments
        def objective(self, I_robust):
            objective_value, delta_F_mean, delta_s_mean, delta_F_stdev, delta_s_stdev, delta_F_array, delta_s_array  = self.calculate_objective(I_robust)
            self.sub_iteration_counter += 1  # Increment the counter
            if self.custom_iteration:
                print("\n\nCurrent Iteration:", self.iteration_counter)
                print("\nCurrent Sub Iteration:", self.sub_iteration_counter)
                # Print current
                print(f"\nObjective Func: {objective_value}") 
                print(f"\nI_array: {np.array2string(I_robust, formatter={'float_kind':lambda x: f"{x:.3e}"}, separator=", ")}")
                print(f"\n∆F (Mean): {delta_F_mean}")
                print(f"\n∆s (Mean): {delta_s_mean}")
                print(f"\n∆F (stDev): {delta_F_stdev}")
                print(f"\n∆s (stDev): {delta_s_stdev}\n")
                # Print current Best
                if self.best_solution['I_array'] is not None:
                    print(f"\nObjective Func Current Best: {self.best_solution['objective_value']}") 
                    print(f"\nI_array Current Best: {np.array2string(self.best_solution['I_array'], formatter={'float_kind':lambda x: f"{x:.3e}"}, separator=", ")}")
                    print(f"\n∆F (Mean) Current Best: {self.best_solution['delta_F_mean']}")
                    print(f"\n∆s (Mean) Current Best: {self.best_solution['delta_s_mean']}")
                    print(f"\n∆F (stDev) Current Best: {self.best_solution['delta_F_stdev']}")
                    print(f"\n∆s (stDev) Current Best: {self.best_solution['delta_s_stdev']}")
                else:
                    print("\nCurrent Best: Not yet found")
            # Track Best Solution
            if objective_value < self.best_solution['objective_value']:
                self.best_solution['I_array'] = np.copy(I_robust)
                self.best_solution['R_array'] = (4 * np.copy(I_robust) / np.pi) ** (1/4)
                self.best_solution['objective_value'] = np.copy(objective_value)
                self.best_solution['delta_F_mean'] = np.copy(delta_F_mean)
                self.best_solution['delta_s_mean'] = np.copy(delta_s_mean)
                self.best_solution['delta_F_stdev'] = np.copy(delta_F_stdev)
                self.best_solution['delta_s_stdev'] = np.copy(delta_s_stdev)
                self.best_solution['delta_F_array'] = np.copy(delta_F_array)
                self.best_solution['delta_s_array'] = np.copy(delta_s_array)
                self.generate_plot(np.linspace(1,num_design_vars,num_design_vars), I_robust) # plot new best
                if self.best_solution['I_plot'] != None and self.best_solution['R_plot'] != None and self.show_sub_plots: 
                    self.best_solution['I_plot'].show()
                    self.best_solution['R_plot'].show()
            return objective_value

        # Callback Function to Track Best Solution
        def callback(self, xk, convergence): 
            # Debug Output
            print(f"\nIteration: {self.iteration_counter} Complete!\n")
            print(f"\nConvergence: {convergence}\n")
            self.iteration_counter += 1  # Increment the counter
            self.sub_iteration_counter = 0  # reset
            self.best_solution['fun_per_iteration'].append(np.copy(self.best_solution['objective_value'])) # Track best fun per iteration
            self.best_solution['I_array_per_iteration'].append(np.copy(self.best_solution['I_array'])) # Track best I_array per iteration
            print("Convergence:", convergence)
            if convergence < self.convergence_tol:  # Stop when improvement is very small
                print(f"Stopping early: Convergence ({convergence}) is below threshold!")
                return True  # Returning True tells SciPy to stop optimization

        def convergence_plot(self):
            x = np.arange(1, len(self.best_solution['fun_per_iteration'])+1) # create an array of integers
            y = self.best_solution['fun_per_iteration']
            data_convergence = {'x': x,'y': y}
            conv_plot = go.Figure()
            # Add data trace
            conv_plot.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name='Objective func value'))
            conv_plot.update_layout(
                title='Objective Function Value vs Iterations',
                xaxis_title='Iterations',
                yaxis_title='Overall RMS Error',
                xaxis=dict(range=[1, len(self.best_solution['fun_per_iteration'])]),
                yaxis=dict(range=[self.best_solution['fun_per_iteration'][-1], self.best_solution['fun_per_iteration'][0]]),
                showlegend=False,
                width=500,     
                height=400  
            )
            self.best_solution['convergence_plot'] = conv_plot # update dictionary
        def error_heatmap(self):
            x = self.s_samples.flatten()
            y = self.F_samples.flatten()
            # First plot
            delta_F_array = self.best_solution['delta_F_array']
            # Take standard deviation across rows (uncertainty samples in Fx and MB) for each column (F-s point)
            z = np.std(delta_F_array, axis=0)
            # Choose normalization method
            if np.min(z) < 0 and np.max(z) > 0:
                # Mixed-sign data: use diverging colormap centered at 0
                norm = TwoSlopeNorm(vmin=np.min(z), vcenter=0.0, vmax=np.max(z))
                cmap = 'coolwarm'
            else:
                # One-sided data (all-positive or all-negative): use standard normalization
                norm = Normalize(vmin=np.min(z), vmax=np.max(z))
                cmap = 'viridis'
            plt.tricontourf(x, y, z, levels=100, cmap=cmap, norm=norm)
            plt.colorbar()
            plt.title('delta_F_array')
            plt.xlabel('applied position (m)')
            plt.ylabel('applied force (N)')
            plt.show()
            # Second plot
            delta_s_array = self.best_solution['delta_s_array']
            # Take standard deviation across rows (uncertainty samples in Fx and MB) for each column (F-s point)
            z = np.std(delta_s_array, axis=0)
            # Choose normalization method
            if np.min(z) < 0 and np.max(z) > 0:
                # Mixed-sign data: use diverging colormap centered at 0
                norm = TwoSlopeNorm(vmin=np.min(z), vcenter=0.0, vmax=np.max(z))
                cmap = 'coolwarm'
            else:
                # One-sided data (all-positive or all-negative): use standard normalization
                norm = Normalize(vmin=np.min(z), vmax=np.max(z))
                cmap = 'viridis'
            plt.tricontourf(x, y, z, levels=100, cmap=cmap, norm=norm)
            plt.colorbar()
            plt.title('delta_s_array')
            plt.xlabel('applied position (m)')
            plt.ylabel('applied force (N)')
            plt.show()

        def simulate (self, moment_array):
            self.objective(moment_array)
            self.error_heatmap()
        def constraints(self, I_robust):
            return np.array([0.06 - self.objective(I_robust)])

        def optimize(self):
            bounds = [(I_min, I_max)] * num_design_vars  # I_robust must be nonnegative
            custom_population = np.array([[1.588, 0.9782, 1.546, 3.305, 12.00, 6.289, 4.179, 8.636, 3.424, 6.742,  12.10, 4.875, 8.867, 6.178, 10.08, 3.867, 4.634, 11.37, 1.688, 6.272],
            [2.210880, 1.104332, 2.755743, 4.649022, 10.980722, 12.425893,
            4.902720, 10.769303, 10.569179, 7.274729, 9.944187, 12.458914,
            10.761120, 6.567529, 11.631773, 1.479769, 11.893281, 12.363217,
            12.126309, 8.917918],
           [1.213812, 1.183281, 2.384906, 5.304047, 1.432505, 2.798145,
            3.169585, 9.366012, 9.273498, 4.749876, 11.983659, 5.483808,
            3.348673, 7.862108, 1.327204, 9.502067, 8.359556, 9.382383,
            2.943219, 6.564464],
           [2.181062, 1.804749, 7.355332, 11.831759, 2.774904, 11.514667,
            8.520873, 5.027771, 7.163544, 2.919268, 9.701850, 11.859776,
            5.193412, 9.017428, 6.893048, 4.656998, 2.019817, 7.498678,
            9.036057, 6.263145],
           [4.831370, 2.205824, 3.843447, 7.455100, 6.173616, 11.684818,
            5.419881, 10.321020, 4.232467, 6.821653, 7.280666, 10.206627,
            5.864983, 3.089812, 8.941772, 5.878380, 5.545610, 5.489226,
            7.266507, 12.402628],
           [9.177459, 11.162661, 8.454268, 5.580987, 5.664475, 3.694760,
            7.281257, 8.051284, 3.102631, 2.335603, 6.228528, 10.613538,
            9.967277, 5.380847, 1.360889, 1.671512, 1.380626, 8.885287,
            9.609554, 5.854110],
           [1.459765, 0.933325, 2.016378, 6.273777, 10.763636, 8.282975,
            8.792419, 10.787499, 9.437382, 4.263042, 2.794780, 10.321010,
            10.784800, 0.470610, 1.771900, 6.328784, 5.087794, 8.454827,
            1.049862, 3.370757],
           [1.430104, 1.886108, 7.785996, 8.837627, 11.338595, 3.801951,
            11.362566, 8.382689, 7.434143, 7.364745, 4.694246, 3.853946,
            6.008871, 5.779444, 2.241904, 0.490742, 1.709227, 5.268903,
            11.579219, 9.291147],
           [2.921456, 3.235320, 12.359084, 3.145624, 7.331679, 3.366357,
            4.864082, 9.202656, 10.772170, 11.278725, 9.079394, 3.062687,
            10.406944, 7.519331, 8.521148, 9.417431, 7.209698, 4.417259,
            2.553712, 11.338955],
           [0.532802, 1.282708, 7.979218, 1.796235, 3.104043, 8.210795,
            5.396954, 10.893735, 10.918041, 7.293297, 12.494034, 7.145053,
            0.109897, 11.522274, 8.725717, 4.160556, 4.986083, 11.092764,
            1.483820, 2.367124],])
            result = differential_evolution(
                func=self.objective,
                bounds=bounds,
                strategy='best1bin',   # Fastest strategy
                maxiter=self.max_iter,          # Number of generations
                popsize=10,            # Population size
                tol=self.convergence_tol, # Tolerance for convergence
                mutation=(0.5, 1),     # Mutation factor
                recombination=0.7,     # Recombination rate
                workers=1,            # Use all available CPU cores
                init = custom_population,
                disp=True,             # Display progress
                callback=self.callback      # Called once per generation
            )
            print("Optimization Stopped: ", result.message)
            print("Final Convergence Value:", result.fun)
            print("Number of Iterations:", result.nit)
            print("Number of Function Evaluations:", result.nfev)
            self.convergence_plot()

    class SurrogateModel():
        def __init__(self, method):
            # Initialize the RBF model
            if method == "RBF":
                self.model = RBF(d0=2, poly_degree=1, reg=1e-25)
            elif method == "KRG":
                self.model = KRG()
            elif method == "KPLSK":
                self.model = KPLSK(
                    n_comp=8,  # Improves speed & generalization
                    nugget=1e-4,
                    theta0=[1e-1],  # Better than 0.01
                    hyper_opt="Cobyla",  # Faster and more stable for high-D problems
                )
            else:
                 raise ValueError("Invalid surrogate model type!") 
             # Initialize scaler
            self.scaler = StandardScaler()

        def calculate_samples(self, objective_fun):
            # Define the problem dimensions and data points
            n_dimensions = num_design_vars
            training_multiplier = 60
            n_train = training_multiplier * n_dimensions  # Initial number of training samples
            test_multiplier = 0.2  # Using 20% of training points for testing
            n_test = int(test_multiplier * n_train)  # Number of testing samples
            # Define the input space bounds
            x_limits = np.array([(I_min, I_max)] * n_dimensions)
            # Create an initial Latin Hypercube Sampling for the input space
            sampling = LHS(xlimits=x_limits, criterion='maximin')
            x_train = sampling(n_train)  # Generate initial training points
            #x_test = sampling(n_test)  # Generate test points
            # Evaluate the objective function on each row of X_train
            y_train = np.array([objective_fun(x) for x in x_train]).reshape(-1, 1)
            #y_test = np.array([objective_fun(x) for x in x_test]).reshape(-1, 1)

            '''# **Adaptive Sampling: Find Underrepresented Y-Values**
            num_bins = 15  # Define number of bins for stratification
            hist, bin_edges = np.histogram(y_train, bins=num_bins)
            # Identify bins with **low density**
            if len(hist) > 0:
                min_bin_count = np.mean(hist)  # Change to mean instead of 25th percentile
                underrepresented_bins = np.where(hist < min_bin_count)[0]  # Bins below mean
            breakpoint()
            # Generate new samples **near underrepresented Y-values**
            new_x_samples = []
            new_y_samples = []
            num_new_samples=200
            max_attempts = 5
            if len(underrepresented_bins) > 0:
                for _ in range(num_new_samples):
                    attempts  = 0
                    while attempts < max_attempts:
                        # Select a random bin from underrepresented bins
                        target_bin = np.random.choice(underrepresented_bins)
                        bin_min, bin_max = bin_edges[target_bin], bin_edges[target_bin + 1]
                        # Find X samples whose Y values fall in this bin
                        candidate_indices = np.where((y_train >= bin_min) & (y_train <= bin_max))[0]
                        if len(candidate_indices) > 0:
                            # Pick a random X sample near this Y range and add slight variation
                            perturbation = np.random.normal(scale=0.1, size=n_dimensions) * (I_max - I_min) * 0.05
                            selected_x = x_train[np.random.choice(candidate_indices)] + perturbation
                            selected_x = np.clip(selected_x, I_min, I_max)  # Ensure within bounds
                            # Evaluate new Y value
                            new_y_sample = objective_fun(selected_x)
                            # **Check if new_y_sample is in an underrepresented bin**
                            if (new_y_sample < bin_edges[0] or new_y_sample > bin_edges[-1]) or \
       any(bin_edges[bin_idx] <= new_y_sample < bin_edges[bin_idx + 1] for bin_idx in underrepresented_bins):
                                # **Valid sample: Add it to dataset**
                                new_x_samples.append(selected_x)
                                new_y_samples.append(new_y_sample)
                                break  # Exit while loop immediately
                        attempts += 1  # Increase attempt count
            # Convert to NumPy arrays
            if new_x_samples:
                new_x_samples = np.array(new_x_samples)
                new_y_samples = np.array(new_y_samples).reshape(-1, 1)
            # Append new samples to the training set
            x_train = np.vstack((x_train, new_x_samples))
            y_train = np.vstack((y_train, new_y_samples))'''
            # Save the scaled training and test sets as .npy
            np.save("x_train.npy", x_train)
            #np.save("x_test.npy", x_test)
            np.save("y_train.npy", y_train)
            #np.save("y_test.npy", y_test)
            raise ValueError("Sampling calculations complete! Please re-run code in surrogate model mode")
        def load(self):
            x_train_raw = np.load("x_train.npy")
            #x_test_raw = np.load("x_test.npy")
            self.y_train = np.load("y_train.npy")
            #self.y_test = np.load("y_test.npy")
            self.scaler.fit(x_train_raw)
            self.x_train = self.scaler.transform(x_train_raw)
            #self.x_test = self.scaler.transform(x_test_raw)
            breakpoint()

        def train(self):
            # Train the model
            self.model.set_training_values(self.x_train, self.y_train)
            self.model.train()
            # Predict on the test set
            y_train_pred, y_train_std = self.model.predict_values(self.x_train), self.model.predict_variances(self.x_train)
            y_test_pred, y_test_std = self.model.predict_values(self.x_test), self.model.predict_variances(self.x_test)
            train_error = np.linalg.norm(self.y_train - y_train_pred) / np.linalg.norm(self.y_train)
            print("\ntrain error:", train_error)
            test_error = np.linalg.norm(self.y_test - y_test_pred) / np.linalg.norm(self.y_test)
            print("\ntest error:", test_error)
            #breakpoint()
        def predict(self, x):
            #Predict output, ensuring the input is normalized using the stored scaler before passing to the model
            x_scaled = self.scaler.transform(np.array(x).reshape(1, -1))
            # Predict using the trained model
            prediction = self.model.predict_values(x_scaled).item()  # Extracts single float value
            return prediction


    robustopt = RobustOpt()
    surrogatemodel = SurrogateModel("KPLSK") # instantiate class object
    if calculate_button.value == 1:
        if dropdown_sensor.value == "Robust Optimization":
            if calculation_method == "explicit":
                robustopt.optimize()
            elif calculation_method == "surrogate model":
                surrogatemodel.load()
                surrogatemodel.train()
                robustopt.load_model(surrogatemodel)
                robustopt.optimize()
        elif dropdown_sensor.value == "Robust Simulation":
            if calculation_method == "explicit":
                robustopt.simulate(moment_array)
            elif calculation_method == "surrogate model":
                surrogatemodel.load()
                surrogatemodel.train()
                robustopt.load_model(surrogatemodel)
                robustopt.objective(moment_array)
        elif dropdown_sensor.value == "Surrogate Model Sampling":
            surrogatemodel.calculate_samples(robustopt.objective) # pass objective fucntion to collect samples


    I_plot, I_array, obj_func, R_plot, conv_plot, R_array, delta_F_stdev, delta_s_stdev = robustopt.get_data()


    I_plot_mo =  mo.ui.plotly(I_plot) if I_plot != None else None
    I_array_mo =  mo.md(f"I array: {np.array2string(I_array, formatter={'float_kind':lambda x: f"{x:.3f}"}, separator=", ")}") if I_array.any() else None
    obj_func_mo = mo.md(f"Objective Func: {obj_func}") if obj_func != None else None
    R_plot_mo =  mo.ui.plotly(R_plot) if R_plot != None else None
    conv_plot_mo = mo.ui.plotly(conv_plot) if conv_plot != None else None
    R_array_mo =  mo.md(f"R array: {np.array2string(R_array, formatter={'float_kind':lambda x: f"{x:.3f}"}, separator=", ")}") if R_array.any() else None
    delta_F_stdev_mo = mo.md(f"∆F (stDev): {delta_F_stdev}") if delta_F_stdev != None else None
    delta_s_stdev_mo = mo.md(f"∆s (stDev): {delta_s_stdev}") if delta_s_stdev != None else None

    mo.accordion({r"**Result**": mo.vstack([
        mo.hstack([I_plot_mo, R_plot_mo]),
        obj_func_mo,
        mo.hstack([I_array_mo, R_array_mo]),
        mo.vstack([delta_F_stdev_mo, delta_s_stdev_mo]),
        conv_plot_mo    
       ])
                 })
    return (
        I_array,
        I_array_mo,
        I_plot,
        I_plot_mo,
        R_array,
        R_array_mo,
        R_plot,
        R_plot_mo,
        RobustOpt,
        SurrogateModel,
        conv_plot,
        conv_plot_mo,
        delta_F_stdev,
        delta_F_stdev_mo,
        delta_s_stdev,
        delta_s_stdev_mo,
        obj_func,
        obj_func_mo,
        robustopt,
        surrogatemodel,
    )


if __name__ == "__main__":
    app.run()
