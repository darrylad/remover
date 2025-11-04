import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector, Button, TextBox, CheckButtons
from matplotlib.gridspec import GridSpec
from scipy import signal
import sys
import os
from datetime import datetime

class VibrationDataRemover:
    def __init__(self, csv_path):
        """Initialize the data remover with CSV file"""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        self.csv_path = csv_path
        self.original_data = pd.read_csv(csv_path)
        self.data = self.original_data.copy()
        
        print(f"Loaded: {csv_path}")
        print(f"Columns: {list(self.data.columns)}")
        print(f"Data shape: {self.data.shape}")
        
        # Assume columns are: time, x, y, z (adjust if needed)
        self.time_col = self.data.columns[0]
        self.x_col = self.data.columns[1]
        self.y_col = self.data.columns[2]
        self.z_col = self.data.columns[3]
        
        # For undo functionality
        self.history = []
        
        # Selection range
        self.selected_start = None
        self.selected_end = None
        
        # Scale mode: True = same scale, False = individual scales
        self.same_scale = False
        
        # View mode: 'time' or 'fft'
        self.view_mode = 'time'
        
        # FFT comparison mode: 'current' or 'original'
        self.fft_compare_mode = 'current'
        
        # FFT y-axis scale mode: True = logarithmic, False = linear
        self.fft_log_scale = True
        
        # Cache for FFT data to maintain consistent scaling
        self.fft_cache = {
            'original': None,
            'current': None
        }
        
        # Calculate sampling rate
        self.calculate_sampling_rate()
        
        # Create outputs folder if it doesn't exist
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.setup_plot()
    
    def calculate_sampling_rate(self):
        """Calculate the sampling rate from the data"""
        if len(self.data) > 1:
            time_diffs = np.diff(self.data[self.time_col].values)
            avg_interval = np.mean(time_diffs)
            self.sampling_rate = 1.0 / avg_interval if avg_interval > 0 else 1000.0
        else:
            self.sampling_rate = 1000.0  # Default fallback
        print(f"Sampling rate: {self.sampling_rate:.2f} Hz")
    
    def setup_plot(self):
        """Create the interactive plot interface"""
        # Create figure with GridSpec for layout
        self.fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(4, 3, figure=self.fig, height_ratios=[3, 3, 3, 0.5])
        
        # Three subplots for x, y, z
        self.ax_x = self.fig.add_subplot(gs[0, :])
        self.ax_y = self.fig.add_subplot(gs[1, :], sharex=self.ax_x)
        self.ax_z = self.fig.add_subplot(gs[2, :], sharex=self.ax_x)
        
        # Control panel area
        self.ax_controls = self.fig.add_subplot(gs[3, :])
        self.ax_controls.axis('off')
        
        # Add SpanSelector to all three plots (BEFORE plotting)
        self.span_x = SpanSelector(
            self.ax_x,
            self.on_select,
            'horizontal',
            useblit=True,
            props=dict(alpha=0.3, facecolor='red'),
            interactive=True,
            drag_from_anywhere=True
        )
        
        self.span_y = SpanSelector(
            self.ax_y,
            self.on_select,
            'horizontal',
            useblit=True,
            props=dict(alpha=0.3, facecolor='red'),
            interactive=True,
            drag_from_anywhere=True
        )
        
        self.span_z = SpanSelector(
            self.ax_z,
            self.on_select,
            'horizontal',
            useblit=True,
            props=dict(alpha=0.3, facecolor='red'),
            interactive=True,
            drag_from_anywhere=True
        )
        
        # Store all span selectors for synchronization
        self.span_selectors = [self.span_x, self.span_y, self.span_z]
        
        # NOW plot the data (after span_selectors exists)
        self.plot_data()
        
        # Add control widgets
        self.setup_controls()
        
        plt.tight_layout()
        plt.show()
    
    def plot_data(self):
        """Plot the data based on current view mode"""
        if self.view_mode == 'time':
            self.plot_time_domain()
        else:
            self.plot_fft()
    
    def plot_time_domain(self):
        """Plot the three acceleration axes in time domain"""
        # Clear previous plots
        self.ax_x.clear()
        self.ax_y.clear()
        self.ax_z.clear()
        
        time = self.data[self.time_col]
        x = self.data[self.x_col]
        y = self.data[self.y_col]
        z = self.data[self.z_col]
        
        # Plot each axis
        self.ax_x.plot(time, x, 'b-', linewidth=0.5)
        self.ax_y.plot(time, y, 'g-', linewidth=0.5)
        self.ax_z.plot(time, z, 'r-', linewidth=0.5)
        
        # Set y-scale based on mode
        if self.same_scale:
            # Same scale for all plots
            all_values = pd.concat([x, y, z])
            y_min, y_max = all_values.min(), all_values.max()
            margin = (y_max - y_min) * 0.1
            
            self.ax_x.set_ylim(y_min - margin, y_max + margin)
            self.ax_y.set_ylim(y_min - margin, y_max + margin)
            self.ax_z.set_ylim(y_min - margin, y_max + margin)
        else:
            # Individual scales for best fit
            for ax, data_series in [(self.ax_x, x), (self.ax_y, y), (self.ax_z, z)]:
                y_min, y_max = data_series.min(), data_series.max()
                margin = (y_max - y_min) * 0.1
                ax.set_ylim(y_min - margin, y_max + margin)
        
        # Labels
        self.ax_x.set_ylabel('X Acceleration', fontsize=10)
        self.ax_y.set_ylabel('Y Acceleration', fontsize=10)
        self.ax_z.set_ylabel('Z Acceleration', fontsize=10)
        self.ax_z.set_xlabel('Time', fontsize=10)
        
        scale_mode = "Same Scale" if self.same_scale else "Individual Scales"
        self.ax_x.set_title(f'X-Axis Vibration ({scale_mode})', fontsize=11)
        self.ax_y.set_title('Y-Axis Vibration', fontsize=11)
        self.ax_z.set_title('Z-Axis Vibration', fontsize=11)
        
        self.ax_x.grid(True, alpha=0.3)
        self.ax_y.grid(True, alpha=0.3)
        self.ax_z.grid(True, alpha=0.3)
        
        # Enable span selectors
        for span in self.span_selectors:
            span.set_active(True)
        
        self.fig.canvas.draw_idle()
    
    def compute_fft_data(self, data_source):
        """Compute FFT data for given data source"""
        if data_source == 'original':
            data_to_use = self.original_data
        else:
            data_to_use = self.data
        
        x = data_to_use[self.x_col].values
        y = data_to_use[self.y_col].values
        z = data_to_use[self.z_col].values
        
        # Calculate Welch's PSD for each axis
        nperseg = min(256, len(x) // 4)
        
        freq_x, psd_x = signal.welch(x, fs=self.sampling_rate, nperseg=nperseg)
        freq_y, psd_y = signal.welch(y, fs=self.sampling_rate, nperseg=nperseg)
        freq_z, psd_z = signal.welch(z, fs=self.sampling_rate, nperseg=nperseg)
        
        return {
            'x': (freq_x, psd_x),
            'y': (freq_y, psd_y),
            'z': (freq_z, psd_z)
        }
    
    def plot_fft(self):
        """Plot Welch's power spectral density for all three axes"""
        # Clear previous plots
        self.ax_x.clear()
        self.ax_y.clear()
        self.ax_z.clear()
        
        # Compute or retrieve FFT data for both original and current
        if self.fft_cache['original'] is None:
            self.fft_cache['original'] = self.compute_fft_data('original')
        
        # Always recompute current data as it may have changed
        self.fft_cache['current'] = self.compute_fft_data('current')
        
        # Get data to display based on mode
        if self.fft_compare_mode == 'original':
            display_data = self.fft_cache['original']
            title_suffix = " (Original Data)"
        else:
            display_data = self.fft_cache['current']
            title_suffix = " (Current Data)"
        
        freq_x, psd_x = display_data['x']
        freq_y, psd_y = display_data['y']
        freq_z, psd_z = display_data['z']
        
        # Plot PSD based on scale mode
        if self.fft_log_scale:
            self.ax_x.semilogy(freq_x, psd_x, 'b-', linewidth=1)
            self.ax_y.semilogy(freq_y, psd_y, 'g-', linewidth=1)
            self.ax_z.semilogy(freq_z, psd_z, 'r-', linewidth=1)
        else:
            self.ax_x.plot(freq_x, psd_x, 'b-', linewidth=1)
            self.ax_y.plot(freq_y, psd_y, 'g-', linewidth=1)
            self.ax_z.plot(freq_z, psd_z, 'r-', linewidth=1)
        
        # Calculate scales based on both original and current data
        orig_data = self.fft_cache['original']
        curr_data = self.fft_cache['current']
        
        if self.same_scale:
            # Same scale across all axes - find global max/min
            all_psd_values = []
            for axis in ['x', 'y', 'z']:
                all_psd_values.extend(orig_data[axis][1])
                all_psd_values.extend(curr_data[axis][1])
            
            y_min = min(all_psd_values)
            y_max = max(all_psd_values)
            
            if self.fft_log_scale:
                # Add margin in log space
                log_range = np.log10(y_max) - np.log10(y_min)
                margin = log_range * 0.1
                y_min_plot = 10 ** (np.log10(y_min) - margin)
                y_max_plot = 10 ** (np.log10(y_max) + margin)
            else:
                # Add margin in linear space
                margin = (y_max - y_min) * 0.1
                y_min_plot = y_min - margin
                y_max_plot = y_max + margin
            
            self.ax_x.set_ylim(y_min_plot, y_max_plot)
            self.ax_y.set_ylim(y_min_plot, y_max_plot)
            self.ax_z.set_ylim(y_min_plot, y_max_plot)
        else:
            # Individual scales - each axis based on its own original and current max/min
            for ax, axis_name in [(self.ax_x, 'x'), (self.ax_y, 'y'), (self.ax_z, 'z')]:
                axis_psd_values = list(orig_data[axis_name][1]) + list(curr_data[axis_name][1])
                y_min = min(axis_psd_values)
                y_max = max(axis_psd_values)
                
                if self.fft_log_scale:
                    # Add margin in log space
                    log_range = np.log10(y_max) - np.log10(y_min)
                    margin = log_range * 0.1
                    y_min_plot = 10 ** (np.log10(y_min) - margin)
                    y_max_plot = 10 ** (np.log10(y_max) + margin)
                else:
                    # Add margin in linear space
                    margin = (y_max - y_min) * 0.1
                    y_min_plot = y_min - margin
                    y_max_plot = y_max + margin
                
                ax.set_ylim(y_min_plot, y_max_plot)
        
        # Labels
        self.ax_x.set_ylabel('PSD [V²/Hz]', fontsize=10)
        self.ax_y.set_ylabel('PSD [V²/Hz]', fontsize=10)
        self.ax_z.set_ylabel('PSD [V²/Hz]', fontsize=10)
        self.ax_z.set_xlabel('Frequency [Hz]', fontsize=10)
        
        scale_mode = "Same Scale" if self.same_scale else "Individual Scales"
        y_scale_mode = "Log" if self.fft_log_scale else "Linear"
        self.ax_x.set_title(f'X-Axis FFT (Welch){title_suffix} ({scale_mode}, {y_scale_mode})', fontsize=11)
        self.ax_y.set_title(f'Y-Axis FFT (Welch){title_suffix}', fontsize=11)
        self.ax_z.set_title(f'Z-Axis FFT (Welch){title_suffix}', fontsize=11)
        
        self.ax_x.grid(True, alpha=0.3)
        self.ax_y.grid(True, alpha=0.3)
        self.ax_z.grid(True, alpha=0.3)
        
        # Disable span selectors in FFT mode
        for span in self.span_selectors:
            span.set_active(False)
        
        self.fig.canvas.draw_idle()
    
    def setup_controls(self):
        """Setup control buttons and text boxes"""
        # Button positions (left, bottom, width, height)
        btn_remove_ax = plt.axes([0.02, 0.02, 0.06, 0.04])
        btn_undo_ax = plt.axes([0.09, 0.02, 0.06, 0.04])
        btn_save_ax = plt.axes([0.16, 0.02, 0.06, 0.04])
        btn_reset_ax = plt.axes([0.23, 0.02, 0.06, 0.04])
        btn_view_ax = plt.axes([0.30, 0.02, 0.08, 0.04])
        
        # FFT comparison button (only visible in FFT mode)
        btn_fft_compare_ax = plt.axes([0.39, 0.02, 0.08, 0.04])
        
        # Checkbox for scale toggle
        checkbox_ax = plt.axes([0.48, 0.02, 0.1, 0.04])
        
        # Checkbox for FFT log/linear toggle (only visible in FFT mode)
        checkbox_fft_scale_ax = plt.axes([0.59, 0.02, 0.08, 0.04])
        
        # Text boxes for precise input
        txt_start_ax = plt.axes([0.75, 0.02, 0.1, 0.04])
        txt_end_ax = plt.axes([0.88, 0.02, 0.1, 0.04])
        
        # Create buttons
        self.btn_remove = Button(btn_remove_ax, 'Remove', color='lightcoral')
        self.btn_undo = Button(btn_undo_ax, 'Undo', color='lightyellow')
        self.btn_save = Button(btn_save_ax, 'Save', color='lightgreen')
        self.btn_reset = Button(btn_reset_ax, 'Reset', color='lightblue')
        self.btn_view = Button(btn_view_ax, 'FFT View', color='lightcyan')
        self.btn_fft_compare = Button(btn_fft_compare_ax, 'Original', color='lightsalmon')
        
        # Create text boxes
        self.txt_start = TextBox(txt_start_ax, 'Start: ', initial='')
        self.txt_end = TextBox(txt_end_ax, 'End: ', initial='')
        
        # Create checkboxes
        self.check_scale = CheckButtons(checkbox_ax, ['Same Scale'], [self.same_scale])
        self.check_fft_scale = CheckButtons(checkbox_fft_scale_ax, ['Log Scale'], [self.fft_log_scale])
        
        # Connect callbacks
        self.btn_remove.on_clicked(self.remove_range)
        self.btn_undo.on_clicked(self.undo_removal)
        self.btn_save.on_clicked(self.save_data)
        self.btn_reset.on_clicked(self.reset_data)
        self.btn_view.on_clicked(self.toggle_view)
        self.btn_fft_compare.on_clicked(self.toggle_fft_compare)
        self.txt_start.on_submit(self.update_start)
        self.txt_end.on_submit(self.update_end)
        self.check_scale.on_clicked(self.toggle_scale)
        self.check_fft_scale.on_clicked(self.toggle_fft_scale)
        
        # Update button visibility
        self.update_button_visibility()
    
    def update_button_visibility(self):
        """Update button visibility and labels based on view mode"""
        if self.view_mode == 'time':
            self.btn_view.label.set_text('FFT View')
            self.btn_fft_compare.ax.set_visible(False)
            self.check_fft_scale.ax.set_visible(False)
            self.btn_remove.ax.set_visible(True)
            self.txt_start.ax.set_visible(True)
            self.txt_end.ax.set_visible(True)
            
            # Disable FFT-only widgets by setting their event connections inactive
            self.btn_fft_compare.active = False
            
            # Enable time-domain widgets
            self.btn_remove.active = True
        else:
            self.btn_view.label.set_text('Time View')
            self.btn_fft_compare.ax.set_visible(True)
            self.check_fft_scale.ax.set_visible(True)
            self.btn_remove.ax.set_visible(False)
            self.txt_start.ax.set_visible(False)
            self.txt_end.ax.set_visible(False)
            
            # Enable FFT-only widgets
            self.btn_fft_compare.active = True
            
            # Disable time-domain widgets
            self.btn_remove.active = False
            
            # Update FFT compare button label
            if self.fft_compare_mode == 'current':
                self.btn_fft_compare.label.set_text('Original')
            else:
                self.btn_fft_compare.label.set_text('Current')
        
        self.fig.canvas.draw_idle()
    
    def toggle_view(self, event):
        """Toggle between time domain and FFT view"""
        if self.view_mode == 'time':
            self.view_mode = 'fft'
            self.fft_compare_mode = 'current'  # Reset to current when entering FFT mode
            print("Switched to FFT view (Welch's method)")
        else:
            self.view_mode = 'time'
            print("Switched to time domain view")
        
        self.update_button_visibility()
        self.plot_data()
    
    def toggle_fft_compare(self, event):
        """Toggle between current and original data in FFT view"""
        if self.fft_compare_mode == 'current':
            self.fft_compare_mode = 'original'
            print("Showing FFT of original data")
        else:
            self.fft_compare_mode = 'current'
            print("Showing FFT of current data")
        
        self.update_button_visibility()
        self.plot_data()
    
    def toggle_scale(self, label):
        """Toggle between same scale and individual scales"""
        self.same_scale = not self.same_scale
        mode = "same scale" if self.same_scale else "individual scales"
        print(f"Scale mode: {mode}")
        self.plot_data()
    
    def toggle_fft_scale(self, label):
        """Toggle between logarithmic and linear scale for FFT"""
        self.fft_log_scale = not self.fft_log_scale
        scale_type = "logarithmic" if self.fft_log_scale else "linear"
        print(f"FFT Y-axis scale: {scale_type}")
        self.plot_data()
    
    def on_select(self, xmin, xmax):
        """Callback when range is selected with SpanSelector"""
        if self.view_mode != 'time':
            return
        
        self.selected_start = xmin
        self.selected_end = xmax
        self.txt_start.set_val(f'{xmin:.4f}')
        self.txt_end.set_val(f'{xmax:.4f}')
        print(f"Selected range: {xmin:.4f} to {xmax:.4f}")
        
        # Synchronize selection across all span selectors
        for span in self.span_selectors:
            if span.extents != (xmin, xmax):
                span.extents = (xmin, xmax)
        
        # Force redraw
        self.fig.canvas.draw_idle()
    
    def update_start(self, text):
        """Update start time from text box"""
        try:
            self.selected_start = float(text)
            # Update span selectors if both start and end are set
            if self.selected_end is not None:
                self.sync_span_selectors(self.selected_start, self.selected_end)
        except ValueError:
            print("Invalid start time")
    
    def update_end(self, text):
        """Update end time from text box"""
        try:
            self.selected_end = float(text)
            # Update span selectors if both start and end are set
            if self.selected_start is not None:
                self.sync_span_selectors(self.selected_start, self.selected_end)
        except ValueError:
            print("Invalid end time")
    
    def sync_span_selectors(self, xmin, xmax):
        """Synchronize all span selectors to the same range"""
        for span in self.span_selectors:
            span.extents = (xmin, xmax)
        self.fig.canvas.draw_idle()
    
    def find_nearest_index(self, time_value):
        """Find the index of the nearest actual data point to given time value"""
        time_array = self.data[self.time_col].values
        idx = np.abs(time_array - time_value).argmin()
        return self.data.index[idx]
    
    def remove_range(self, event):
        """Remove selected time range and close gaps"""
        if self.selected_start is None or self.selected_end is None:
            print("Please select a range first!")
            return
        
        start = min(self.selected_start, self.selected_end)
        end = max(self.selected_start, self.selected_end)
        
        # Find nearest actual data points
        start_idx = self.find_nearest_index(start)
        end_idx = self.find_nearest_index(end)
        
        # Get actual time values from the data
        actual_start = self.data.loc[start_idx, self.time_col]
        actual_end = self.data.loc[end_idx, self.time_col]
        
        print(f"Selected: {start:.4f} to {end:.4f}")
        print(f"Nearest actual points: {actual_start:.6f} to {actual_end:.6f}")
        
        # Save current state for undo
        self.history.append(self.data.copy())
        
        # Find indices in range (inclusive of start and end points)
        mask = (self.data[self.time_col] >= actual_start) & (self.data[self.time_col] <= actual_end)
        indices_to_remove = self.data[mask].index
        
        if len(indices_to_remove) == 0:
            print("No data in selected range")
            return
        
        # Calculate exact time gap from actual data points
        time_gap = actual_end - actual_start
        
        # Get the index just after the removed range for adjustment reference
        last_removed_idx = indices_to_remove[-1]
        
        # Remove the range
        self.data = self.data[~mask].copy()
        
        # Adjust all subsequent timestamps to maintain even spacing
        # All points after the removed range are shifted back by the time_gap
        subsequent_mask = self.data.index > last_removed_idx
        self.data.loc[subsequent_mask, self.time_col] -= time_gap
        
        # Reset index
        self.data.reset_index(drop=True, inplace=True)
        
        print(f"Removed {len(indices_to_remove)} points")
        print(f"Time gap closed: {time_gap:.6f} (based on actual data points)")
        
        # Recalculate sampling rate and invalidate FFT cache
        self.calculate_sampling_rate()
        self.fft_cache['current'] = None  # Invalidate current FFT cache
        
        # Clear selection
        self.selected_start = None
        self.selected_end = None
        self.txt_start.set_val('')
        self.txt_end.set_val('')
        
        # Clear all span selectors
        for span in self.span_selectors:
            span.extents = (0, 0)
        
        # Replot
        self.plot_data()
    
    def undo_removal(self, event):
        """Undo last removal"""
        if len(self.history) > 0:
            self.data = self.history.pop()
            print("Undone last removal")
            self.calculate_sampling_rate()
            self.fft_cache['current'] = None  # Invalidate current FFT cache
            self.plot_data()
        else:
            print("Nothing to undo")
    
    def reset_data(self, event):
        """Reset to original data"""
        self.data = self.original_data.copy()
        self.history = []
        print("Reset to original data")
        self.calculate_sampling_rate()
        self.fft_cache['current'] = None  # Invalidate current FFT cache
        self.plot_data()
    
    def save_data(self, event):
        """Save cleaned data to outputs folder"""
        # Generate new filename with timestamp
        base_name = os.path.splitext(os.path.basename(self.csv_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{base_name}_cleaned_{timestamp}.csv"
        output_path = os.path.join(self.output_dir, output_filename)
        
        self.data.to_csv(output_path, index=False)
        print(f"✅ Saved cleaned data to: {output_path}")
        print(f"Original file unchanged: {self.csv_path}")
        print("Timestamps have been adjusted in the saved file")


def main():
    """Main function to handle file input"""
    # Check if file provided as command line argument
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        # Prompt for file path
        print("=" * 60)
        print("Vibration Data Anomaly Remover")
        print("=" * 60)
        csv_file = input("\nEnter the path to your CSV file: ").strip()
        
        # Remove quotes if user pasted path with quotes
        csv_file = csv_file.strip('"').strip("'")
    
    try:
        remover = VibrationDataRemover(csv_file)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease check the file path and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()