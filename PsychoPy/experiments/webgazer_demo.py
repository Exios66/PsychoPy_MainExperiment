#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WebGazer Demo Experiment

This experiment demonstrates the integration of PsychoPy with WebGazer.js
for web-based eye tracking. It includes a simple visual task with real-time
gaze visualization and data collection.

Features:
- WebGazer.js integration for webcam-based eye tracking
- Calibration and validation procedures
- Real-time gaze visualization
- Data collection and analysis
- Example visual task with gaze-contingent display

Usage:
    python -m PsychoPy.experiments.webgazer_demo
"""

import os
import json
import time
import webbrowser
from pathlib import Path
from datetime import datetime

from psychopy import visual, core, event, gui, logging
import matplotlib.pyplot as plt
import numpy as np

from ..utils import WebGazerBridge, analyze_session
from ..config import DATA_DIR


class WebGazerDemo:
    """Demo experiment using WebGazer.js for eye tracking."""

    def __init__(self, port=8765, fullscreen=False, debug=False):
        """
        Initialize the experiment.

        Parameters
        ----------
        port : int, optional
            The port for the WebGazer bridge. Default is 8765.
        fullscreen : bool, optional
            Whether to run in fullscreen mode. Default is False.
        debug : bool, optional
            Whether to enable debug mode. Default is False.
        """
        # Configure logging
        log_level = logging.DEBUG if debug else logging.INFO
        logging.console.setLevel(log_level)
        
        # Initialize experiment components
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data_dir = DATA_DIR / self.session_id
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize WebGazer bridge
        self.bridge = WebGazerBridge(
            session_id=self.session_id,
            port=port,
            save_locally=True
        )
        
        # Initialize PsychoPy window
        self.win = visual.Window(
            size=(1024, 768),
            fullscr=fullscreen,
            monitor="testMonitor",
            units="pix",
            color=(0.9, 0.9, 0.9),
            winType='pyglet'
        )
        
        # Initialize stimuli
        self.instructions = visual.TextStim(
            self.win,
            text="WebGazer.js Demo Experiment\n\n"
                 "This experiment demonstrates webcam-based eye tracking.\n\n"
                 "You will see a series of targets on the screen.\n"
                 "Look at each target as it appears.\n\n"
                 "Press SPACE to begin calibration.",
            color='black',
            height=24,
            wrapWidth=800
        )
        
        self.calibration_instructions = visual.TextStim(
            self.win,
            text="Calibration\n\n"
                 "Look at each target as it appears and click on it.\n"
                 "Try to keep your head still during calibration.",
            color='black',
            height=24,
            wrapWidth=800
        )
        
        self.validation_instructions = visual.TextStim(
            self.win,
            text="Validation\n\n"
                 "Look at each target as it appears.\n"
                 "Do not click during validation.",
            color='black',
            height=24,
            wrapWidth=800
        )
        
        self.target = visual.Circle(
            self.win,
            radius=15,
            fillColor='red',
            lineColor='black',
            lineWidth=2
        )
        
        self.gaze_dot = visual.Circle(
            self.win,
            radius=5,
            fillColor='blue',
            lineColor=None,
            opacity=0.5,
            autoLog=False
        )
        
        self.feedback = visual.TextStim(
            self.win,
            text="",
            color='black',
            height=24,
            pos=(0, -300),
            wrapWidth=800
        )
        
        # Initialize experiment variables
        self.current_gaze = {'x': 0, 'y': 0, 'valid': False}
        self.calibration_points = []
        self.validation_results = None
        self.trial_data = []
        self.experiment_active = False

    def on_gaze_data(self, data):
        """
        Callback function for gaze data.

        Parameters
        ----------
        data : dict
            The gaze data from WebGazer.js.
        """
        if 'x' in data and 'y' in data:
            # Convert from normalized coordinates to pixel coordinates
            x = (data['x'] * self.win.size[0]) - (self.win.size[0] / 2)
            y = -((data['y'] * self.win.size[1]) - (self.win.size[1] / 2))
            
            # Update current gaze position
            self.current_gaze = {
                'x': x,
                'y': y,
                'timestamp': data.get('timestamp', time.time()),
                'valid': True
            }
            
            # Update gaze dot position if experiment is active
            if self.experiment_active and hasattr(self, 'gaze_dot'):
                self.gaze_dot.pos = (x, y)
                self.gaze_dot.draw()
                
            # Log gaze data for the current trial
            if hasattr(self, 'current_trial') and self.current_trial is not None:
                self.trial_data.append({
                    'trial': self.current_trial,
                    'timestamp': data.get('timestamp', time.time()),
                    'x': x,
                    'y': y,
                    'target_x': self.target.pos[0] if hasattr(self, 'target') else None,
                    'target_y': self.target.pos[1] if hasattr(self, 'target') else None
                })

    def run(self):
        """Run the experiment."""
        try:
            # Show instructions
            self.instructions.draw()
            self.win.flip()
            event.waitKeys(keyList=['space'])
            
            # Start WebGazer bridge
            self.bridge.start()
            self.bridge.register_gaze_callback(self.on_gaze_data)
            
            # Wait for WebGazer to initialize
            core.wait(2.0)
            
            # Run calibration
            self.run_calibration()
            
            # Run validation
            self.run_validation()
            
            # Run the main experiment
            self.run_experiment()
            
            # Analyze data
            self.analyze_data()
            
        except Exception as e:
            logging.error(f"Error in experiment: {str(e)}")
        finally:
            # Clean up
            self.quit()

    def run_calibration(self):
        """Run the calibration procedure."""
        # Show calibration instructions
        self.calibration_instructions.draw()
        self.win.flip()
        core.wait(3.0)
        
        # Define calibration points (9-point calibration)
        width, height = self.win.size
        points = [
            (-width/4, -height/4), (0, -height/4), (width/4, -height/4),
            (-width/4, 0), (0, 0), (width/4, 0),
            (-width/4, height/4), (0, height/4), (width/4, height/4)
        ]
        
        # Run calibration
        for i, point in enumerate(points):
            # Show target
            self.target.pos = point
            self.target.draw()
            self.win.flip()
            
            # Wait for user to look at target and click
            event.clearEvents()
            while True:
                if event.getKeys(keyList=['escape']):
                    self.quit()
                    return
                
                mouse = event.getMouseButtons()
                if mouse[0]:  # Left button clicked
                    break
                
                core.wait(0.1)
            
            # Store calibration point
            self.calibration_points.append({
                'expected': {'x': point[0], 'y': point[1]},
                'observed': {'x': self.current_gaze['x'], 'y': self.current_gaze['y']}
            })
            
            # Brief pause between points
            core.wait(0.5)
        
        # Send calibration data to WebGazer
        self.bridge.send_message('calibration_data', self.calibration_points)
        
        # Show completion message
        self.feedback.text = "Calibration complete!"
        self.feedback.draw()
        self.win.flip()
        core.wait(2.0)

    def run_validation(self):
        """Run the validation procedure."""
        # Show validation instructions
        self.validation_instructions.draw()
        self.win.flip()
        core.wait(3.0)
        
        # Define validation points (different from calibration points)
        width, height = self.win.size
        points = [
            (-width/3, -height/3), (width/3, -height/3),
            (0, 0),
            (-width/3, height/3), (width/3, height/3)
        ]
        
        # Run validation
        validation_data = []
        for i, point in enumerate(points):
            # Show target
            self.target.pos = point
            self.target.draw()
            self.win.flip()
            
            # Wait for user to look at target
            core.wait(1.0)
            
            # Record gaze position
            validation_data.append({
                'expected': {'x': point[0], 'y': point[1]},
                'observed': {'x': self.current_gaze['x'], 'y': self.current_gaze['y']}
            })
            
            # Brief pause between points
            core.wait(0.5)
        
        # Calculate validation error
        errors = []
        for point in validation_data:
            ex, ey = point['expected']['x'], point['expected']['y']
            ox, oy = point['observed']['x'], point['observed']['y']
            error = np.sqrt((ex - ox)**2 + (ey - oy)**2)
            errors.append(error)
        
        mean_error = np.mean(errors)
        self.validation_results = {
            'points': validation_data,
            'errors': errors,
            'mean_error': mean_error
        }
        
        # Show validation results
        self.feedback.text = f"Validation complete!\nMean error: {mean_error:.2f} pixels"
        self.feedback.draw()
        self.win.flip()
        core.wait(2.0)

    def run_experiment(self):
        """Run the main experiment."""
        # Show experiment instructions
        self.instructions.text = "Main Experiment\n\n" \
                                "You will see a series of targets.\n" \
                                "Look at each target as quickly as possible.\n\n" \
                                "Press SPACE to begin."
        self.instructions.draw()
        self.win.flip()
        event.waitKeys(keyList=['space'])
        
        # Start experiment
        self.experiment_active = True
        
        # Define target positions
        width, height = self.win.size
        positions = [
            (-width/3, -height/3), (width/3, -height/3),
            (-width/3, height/3), (width/3, height/3),
            (0, 0), (-width/2, 0), (width/2, 0),
            (0, -height/2), (0, height/2)
        ]
        
        # Randomize positions
        np.random.shuffle(positions)
        
        # Run trials
        for i, position in enumerate(positions):
            self.current_trial = i + 1
            
            # Show target
            self.target.pos = position
            self.target.draw()
            
            # Start trial timer
            trial_start = time.time()
            
            # Wait for gaze to reach target or timeout
            timeout = 3.0  # seconds
            target_reached = False
            
            while time.time() - trial_start < timeout:
                # Check for escape key
                if event.getKeys(keyList=['escape']):
                    self.quit()
                    return
                
                # Check if gaze is on target
                if self.current_gaze['valid']:
                    dx = self.current_gaze['x'] - position[0]
                    dy = self.current_gaze['y'] - position[1]
                    distance = np.sqrt(dx**2 + dy**2)
                    
                    if distance < 50:  # 50 pixels threshold
                        target_reached = True
                        break
                
                # Update display
                self.target.draw()
                self.gaze_dot.draw()
                self.win.flip()
                
                # Brief pause
                core.wait(0.01)
            
            # Record trial result
            trial_end = time.time()
            trial_duration = trial_end - trial_start
            
            # Brief pause between trials
            core.wait(0.5)
        
        # End experiment
        self.experiment_active = False
        
        # Show completion message
        self.feedback.text = "Experiment complete!\nThank you for participating."
        self.feedback.draw()
        self.win.flip()
        core.wait(3.0)

    def analyze_data(self):
        """Analyze the collected data."""
        # Save trial data
        data_file = self.data_dir / 'trial_data.json'
        with open(data_file, 'w') as f:
            json.dump(self.trial_data, f, indent=2)
        
        # Save validation results
        if self.validation_results:
            validation_file = self.data_dir / 'validation_results.json'
            with open(validation_file, 'w') as f:
                json.dump(self.validation_results, f, indent=2)
        
        # Generate basic visualizations
        if len(self.trial_data) > 0:
            try:
                # Create a gaze plot
                plt.figure(figsize=(10, 8))
                
                # Plot gaze points
                x_coords = [d['x'] for d in self.trial_data]
                y_coords = [d['y'] for d in self.trial_data]
                plt.scatter(x_coords, y_coords, s=1, alpha=0.5, c='blue')
                
                # Plot targets
                target_x = [d['target_x'] for d in self.trial_data if d['target_x'] is not None]
                target_y = [d['target_y'] for d in self.trial_data if d['target_y'] is not None]
                plt.scatter(target_x, target_y, s=50, c='red', marker='o')
                
                # Set plot properties
                plt.title('Gaze Plot')
                plt.xlabel('X Position (pixels)')
                plt.ylabel('Y Position (pixels)')
                plt.grid(True, alpha=0.3)
                
                # Save plot
                plot_file = self.data_dir / 'gaze_plot.png'
                plt.savefig(plot_file)
                plt.close()
                
                # Open the plot file
                webbrowser.open(f'file://{plot_file.absolute()}')
                
            except Exception as e:
                logging.error(f"Error generating visualizations: {str(e)}")

    def quit(self):
        """Clean up and quit the experiment."""
        # Stop WebGazer bridge
        if hasattr(self, 'bridge'):
            self.bridge.stop()
        
        # Close PsychoPy window
        if hasattr(self, 'win'):
            self.win.close()
        
        # Quit PsychoPy
        core.quit()


def main():
    """Run the WebGazer demo experiment."""
    # Show experiment dialog
    dlg = gui.Dlg(title="WebGazer Demo")
    dlg.addText("WebGazer.js Demo Experiment")
    dlg.addField("Participant ID:", "test")
    dlg.addField("Fullscreen:", False)
    dlg.addField("Debug mode:", False)
    
    ok_data = dlg.show()
    if not dlg.OK:
        return
    
    # Get dialog data
    participant_id, fullscreen, debug = ok_data
    
    # Run experiment
    experiment = WebGazerDemo(
        fullscreen=fullscreen,
        debug=debug
    )
    experiment.run()


if __name__ == "__main__":
    main() 