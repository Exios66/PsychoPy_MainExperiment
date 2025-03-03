# PsychoPy WebGazer Integration

This project integrates WebGazer.js with PsychoPy to enable web-based eye tracking in psychological experiments. It provides a comprehensive framework for creating, running, and analyzing eye-tracking experiments using PsychoPy and WebGazer.js.

## Overview

WebGazer.js is a JavaScript library that uses computer vision techniques to enable eye tracking through a standard webcam. This project integrates WebGazer.js with PsychoPy, a powerful Python library for creating psychology experiments, to provide a complete solution for web-based eye tracking.

Key features:

- Web-based eye tracking using WebGazer.js
- Integration with PsychoPy for experiment design and control
- Real-time visualization of gaze data
- Comprehensive data collection and analysis tools
- Cross-platform compatibility (Windows, macOS, Linux)

## Installation

### Prerequisites

- Python 3.9 or later
- PsychoPy 2023.1.0 or later
- A modern web browser (Chrome or Firefox recommended)
- A webcam

### Setup

1. Clone this repository:

```bash
git clone https://github.com/yourusername/PsychoPy_MainExperiment.git
cd PsychoPy_MainExperiment
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

1. Activate the virtual environment:

```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Run the application:

```bash
python PsychoPy/run_application.py
```

3. Open a web browser and navigate to `http://localhost:5000` to access the web interface.

### Creating an Experiment

1. Create a new experiment file in the `PsychoPy/experiments` directory.
2. Import the necessary modules:

```python
from PsychoPy.utils.webgazer_bridge import WebGazerBridge
```

3. Initialize the WebGazerBridge in your experiment:

```python
bridge = WebGazerBridge(session_id='my_experiment')
bridge.start()
```

4. Use the bridge to collect gaze data:

```python
bridge.register_gaze_callback(my_callback_function)
```

5. Clean up when done:

```python
bridge.stop()
```

### Example Experiment

Here's a simple example of a PsychoPy experiment that uses WebGazer.js for eye tracking:

```python
from psychopy import visual, core
from PsychoPy.utils.webgazer_bridge import WebGazerBridge

# Initialize PsychoPy window
win = visual.Window([800, 600], fullscr=False, monitor="testMonitor", units="pix")

# Initialize WebGazerBridge
bridge = WebGazerBridge(session_id='example_experiment')
bridge.start()

# Create a stimulus
stim = visual.TextStim(win, text="Look at this text", pos=[0, 0])

# Define a callback function for gaze data
def gaze_callback(data):
    print(f"Gaze position: {data['x']}, {data['y']}")

# Register the callback
bridge.register_gaze_callback(gaze_callback)

# Run the experiment
stim.draw()
win.flip()
core.wait(5)  # Wait for 5 seconds

# Clean up
bridge.stop()
win.close()
```

## WebGazer.js Integration

WebGazer.js is integrated with PsychoPy through a WebSocket connection. The `WebGazerBridge` class in `PsychoPy/utils/webgazer_bridge.py` handles the communication between PsychoPy and WebGazer.js.

### Key Components

1. **WebGazerBridge**: Handles the communication between PsychoPy and WebGazer.js.
2. **webgazer.js**: The main WebGazer.js library.
3. **webgazer_integration.js**: A custom JavaScript file that integrates WebGazer.js with the web interface.
4. **webgazer_integration.html**: An HTML file that demonstrates the integration of WebGazer.js.

### Calibration

WebGazer.js requires calibration before it can accurately track eye movements. The calibration process involves looking at a series of points on the screen. The `WebGazerBridge` class provides methods for running the calibration process:

```python
bridge = WebGazerBridge()
bridge.start()
bridge.send_message('calibrate', {'points': 9})  # Run a 9-point calibration
```

### Data Collection

Gaze data is collected through the WebSocket connection and can be accessed in PsychoPy through callback functions:

```python
def gaze_callback(data):
    print(f"Gaze position: {data['x']}, {data['y']}")

bridge.register_gaze_callback(gaze_callback)
```

## Advanced Features

### Custom Calibration

You can customize the calibration process by modifying the `calibration.py` file in the `PsychoPy/Scripts` directory. This file contains the code for the calibration process, including the number and position of calibration points.

### Data Analysis

The `PsychoPy/utils/analysis.py` file contains functions for analyzing gaze data, including fixation detection, saccade detection, and heatmap generation.

### Visualization

The web interface provides real-time visualization of gaze data, including heatmaps and scanpaths. You can customize the visualization by modifying the `PsychoPy/templates/index.html` file.

## Troubleshooting

### Common Issues

1. **WebGazer.js not loading**: Make sure that the WebGazer.js library is properly included in the HTML file and that the path is correct.

2. **Calibration not working**: Make sure that the webcam is properly connected and that the browser has permission to access it.

3. **Gaze data not being received**: Check the WebSocket connection and make sure that the WebGazerBridge is properly initialized and started.

### Debugging

You can enable debug logging to get more information about what's happening:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [WebGazer.js](https://webgazer.cs.brown.edu/) - The JavaScript library for eye tracking
- [PsychoPy](https://www.psychopy.org/) - The Python library for creating psychology experiments
