# Quick Start Guide

This guide will help you get started with the PsychoPy WebGazer Integration.

## Installation

### Prerequisites

- Python 3.9 or later
- A modern web browser (Chrome or Firefox recommended)
- A webcam

### Installation Steps

#### On Unix-based Systems (macOS, Linux)

1. Clone the repository:

```bash
git clone https://github.com/yourusername/PsychoPy_MainExperiment.git
cd PsychoPy_MainExperiment
```

2. Run the installation script:

```bash
chmod +x install.sh
./install.sh
```

3. Activate the virtual environment:

```bash
source venv/bin/activate
```

#### On Windows

1. Clone the repository:

```bash
git clone https://github.com/yourusername/PsychoPy_MainExperiment.git
cd PsychoPy_MainExperiment
```

2. Run the installation script:

```bash
install.bat
```

3. Activate the virtual environment:

```bash
venv\Scripts\activate.bat
```

## Running the Demo Experiment

1. Make sure the virtual environment is activated.

2. Run the demo experiment:

```bash
python -m PsychoPy.experiments.webgazer_demo
```

3. Follow the on-screen instructions to calibrate and run the experiment.

## Running the Main Application

1. Make sure the virtual environment is activated.

2. Run the main application:

```bash
python PsychoPy/run_application.py
```

3. Open a web browser and navigate to `http://localhost:5000` to access the web interface.

## Creating Your Own Experiment

1. Create a new Python file in the `PsychoPy/experiments` directory.

2. Import the necessary modules:

```python
from psychopy import visual, core, event
from PsychoPy.utils.webgazer_bridge import WebGazerBridge
```

3. Initialize the WebGazerBridge:

```python
bridge = WebGazerBridge(session_id='my_experiment')
bridge.start()
```

4. Register a callback function for gaze data:

```python
def gaze_callback(data):
    print(f"Gaze position: {data['x']}, {data['y']}")

bridge.register_gaze_callback(gaze_callback)
```

5. Create your experiment logic using PsychoPy.

6. Clean up when done:

```python
bridge.stop()
```

## Next Steps

- Read the [WebGazer Integration Documentation](webgazer_integration.md) for more details.
- Check out the [Example Experiment](../PsychoPy/experiments/webgazer_demo.py) for a complete example.
- Explore the [PsychoPy Documentation](https://www.psychopy.org/documentation.html) for more information about PsychoPy.
