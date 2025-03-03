# WebGazer.js Integration with PsychoPy

This document provides detailed information about the integration of WebGazer.js with PsychoPy for eye tracking experiments.

## Overview

WebGazer.js is a JavaScript library that enables eye tracking through a standard webcam. This integration allows PsychoPy experiments to use WebGazer.js for eye tracking, providing a cost-effective alternative to specialized eye tracking hardware.

## Architecture

The integration consists of the following components:

1. **WebGazerBridge**: A Python class that handles communication between PsychoPy and WebGazer.js via WebSockets.
2. **WebGazer.js**: The JavaScript library that performs the eye tracking.
3. **Integration Scripts**: JavaScript and HTML files that integrate WebGazer.js with the web interface.
4. **PsychoPy Experiment**: Python code that uses the WebGazerBridge to collect and analyze gaze data.

## WebGazerBridge

The `WebGazerBridge` class is defined in `PsychoPy/utils/webgazer_bridge.py`. It provides the following functionality:

- Starting and stopping the WebSocket server
- Sending and receiving messages between PsychoPy and WebGazer.js
- Registering callback functions for gaze data
- Saving gaze data to disk

### Key Methods

- `start()`: Starts the WebSocket server.
- `stop()`: Stops the WebSocket server.
- `register_gaze_callback(callback)`: Registers a callback function for gaze data.
- `send_message(message_type, data)`: Sends a message to WebGazer.js.
- `save_client_html()`: Saves the client HTML file for WebGazer.js.

## WebGazer.js Integration

The WebGazer.js library is integrated with PsychoPy through the following files:

- `PsychoPy/Scripts/js/webgazer.js`: The WebGazer.js library.
- `PsychoPy/Scripts/js/webgazer_integration.js`: JavaScript code that integrates WebGazer.js with the web interface.
- `PsychoPy/Scripts/html/webgazer_integration.html`: HTML file that demonstrates the integration of WebGazer.js.

### WebGazer.js Configuration

WebGazer.js can be configured through the `webgazer_integration.js` file. The following options can be modified:

- `webgazer.setTracker()`: Sets the tracking method (e.g., "TFFacemesh").
- `webgazer.setRegression()`: Sets the regression method (e.g., "ridge").
- `webgazer.showVideoPreview()`: Shows or hides the video preview.
- `webgazer.showPredictionPoints()`: Shows or hides the prediction points.

## Calibration and Validation

WebGazer.js requires calibration before it can accurately track eye movements. The calibration process involves looking at a series of points on the screen and clicking on them. The validation process evaluates the accuracy of the calibration.

### Calibration Process

1. The user is shown a series of targets at known positions on the screen.
2. The user looks at each target and clicks on it.
3. WebGazer.js uses the click positions and the corresponding gaze positions to calibrate the eye tracking.

### Validation Process

1. The user is shown a series of targets at known positions on the screen.
2. The user looks at each target without clicking.
3. The system compares the known target positions with the gaze positions reported by WebGazer.js.
4. The system calculates the mean error between the target positions and the gaze positions.

## Data Collection and Analysis

Gaze data is collected through the WebSocket connection and can be accessed in PsychoPy through callback functions. The data can be saved to disk for later analysis.

### Data Format

The gaze data is provided as a dictionary with the following keys:

- `x`: The x-coordinate of the gaze position (normalized to the range [0, 1]).
- `y`: The y-coordinate of the gaze position (normalized to the range [0, 1]).
- `timestamp`: The timestamp of the gaze position.

### Data Analysis

The `PsychoPy/utils/analysis.py` file provides functions for analyzing gaze data, including:

- Fixation detection
- Saccade detection
- Heatmap generation
- Scanpath visualization

## Example Usage

Here's a simple example of how to use the WebGazer.js integration in a PsychoPy experiment:

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

## Performance Considerations

WebGazer.js is a JavaScript library that runs in the browser, which means it has some limitations compared to specialized eye tracking hardware:

- **Accuracy**: WebGazer.js is less accurate than specialized eye tracking hardware. The accuracy can be improved through calibration, but it will still be lower than hardware-based solutions.

- **Sampling Rate**: WebGazer.js has a lower sampling rate than specialized eye tracking hardware. The sampling rate depends on the browser and the hardware, but it is typically around 30 Hz.

- **Latency**: WebGazer.js has higher latency than specialized eye tracking hardware. The latency depends on the browser and the hardware, but it is typically around 100 ms.

## Advanced Features

### Custom Calibration

You can customize the calibration process by modifying the `calibration.py` file in the `PsychoPy/Scripts` directory. This file contains the code for the calibration process, including the number and position of calibration points.

### Custom Regression

WebGazer.js supports different regression methods for mapping gaze positions to screen coordinates. You can set the regression method using the `webgazer.setRegression()` function in the `webgazer_integration.js` file.

### Custom Tracking

WebGazer.js supports different tracking methods for detecting the eyes and estimating gaze positions. You can set the tracking method using the `webgazer.setTracker()` function in the `webgazer_integration.js` file.

## References

- [WebGazer.js](https://webgazer.cs.brown.edu/): The official WebGazer.js website.
- [WebGazer.js GitHub Repository](https://github.com/brownhci/WebGazer): The WebGazer.js GitHub repository.
- [PsychoPy](https://www.psychopy.org/): The official PsychoPy website.
