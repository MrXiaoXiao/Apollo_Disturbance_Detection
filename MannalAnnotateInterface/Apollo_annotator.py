import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import csv
import os
import scipy.signal
import obspy
import sys

class ArrayPlotterApp:
    def __init__(self, window):
        self.window = window
        window.title("Apollo Disturbance Annotator")
        window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.min_freq = 0.1 * 1e-3
        self.max_freq = 10 * 1e-3
        self.min_period = 200
        self.max_period = 1500

        self.button_frame = tk.Frame(window)
        self.button_frame.grid(row=0, column=0, sticky="ew")
        
        self.open_button = tk.Button(self.button_frame, text="Open Data\n(O)", command=self.open_array)
        self.open_button.grid(row=0, column=0, sticky="ew")
    
        self.load_windows_button = tk.Button(self.button_frame, text="Load Windows\n(L)", command=self.load_windows)
        self.load_windows_button.grid(row=0, column=1, sticky="ew")

        self.select_window_button = tk.Button(self.button_frame, text="Select Window\n(A)", command=self.prepare_to_select_window)
        self.select_window_button.grid(row=0, column=2, sticky="ew")
        
        self.stop_choosing_button = tk.Button(self.button_frame, text="Stop Choosing\n(S)", command=self.stop_choosing)
        self.stop_choosing_button.grid(row=0, column=3, sticky="ew")
        
        self.redo_button = tk.Button(self.button_frame, text="Redo\n(R)", command=self.redo_selection)
        self.redo_button.grid(row=0, column=4, sticky="ew")

        self.update_fft_button = tk.Button(self.button_frame, text="Update FFT\n(U)", command=self.update_fft)
        self.update_fft_button.grid(row=0, column=5, sticky="ew")
        
        self.save_windows_button = tk.Button(self.button_frame, text="Save Windows\n(S)", command=self.save_windows)
        self.save_windows_button.grid(row=0, column=6, sticky="ew")
        
        self.edit_window_button = tk.Button(self.button_frame, text="Edit Window\n(E)", command=self.toggle_edit_mode, bg='gray')
        self.edit_window_button.grid(row=0, column=7, sticky="ew")
        
        self.delete_window_button = tk.Button(self.button_frame, text="Delete Window\n(D)", command=self.toggle_delete_mode)
        self.delete_window_button.grid(row=0, column=8, sticky="ew")
        self.deleting_window = False
        
        self.merge_window_button = tk.Button(self.button_frame, text="Merge Window\n(M)", command=self.merge_window)
        self.merge_window_button.grid(row=0, column=9, sticky="ew")
        
        self.zoom_in_button = tk.Button(self.button_frame, text="Zoom In\n(+)", command=self.zoom_in)
        self.zoom_in_button.grid(row=0, column=10, sticky="ew")

        self.zoom_out_button = tk.Button(self.button_frame, text="Zoom Out\n(-)", command=self.zoom_out)
        self.zoom_out_button.grid(row=0, column=11, sticky="ew")

        for i in range(12):
            self.button_frame.columnconfigure(i, weight=1)
        
        self.range_frame = tk.Frame(window)
        self.range_frame.grid(row=1, column=0, sticky="ew", pady=5)
        
        self.min_freq_label = tk.Label(self.range_frame, text="Min Freq (Hz):")
        self.min_freq_label.grid(row=0, column=0, padx=2, pady=2)
        self.min_freq_entry = tk.Entry(self.range_frame, width=10)
        self.min_freq_entry.insert(0, str(self.min_freq))
        self.min_freq_entry.grid(row=0, column=1, padx=2, pady=2)
        
        self.max_freq_label = tk.Label(self.range_frame, text="Max Freq (Hz):")
        self.max_freq_label.grid(row=0, column=2, padx=2, pady=2)
        self.max_freq_entry = tk.Entry(self.range_frame, width=10)
        self.max_freq_entry.insert(0, str(self.max_freq))
        self.max_freq_entry.grid(row=0, column=3, padx=2, pady=2)
        
        self.min_period_label = tk.Label(self.range_frame, text="Min Period (s):")
        self.min_period_label.grid(row=0, column=4, padx=2, pady=2)
        self.min_period_entry = tk.Entry(self.range_frame, width=10)
        self.min_period_entry.insert(0, str(self.min_period))
        self.min_period_entry.grid(row=0, column=5, padx=2, pady=2)

        self.max_period_label = tk.Label(self.range_frame, text="Max Period (s):")
        self.max_period_label.grid(row=0, column=6, padx=2, pady=2)
        self.max_period_entry = tk.Entry(self.range_frame, width=10)
        self.max_period_entry.insert(0, str(self.max_period))
        self.max_period_entry.grid(row=0, column=7, padx=2, pady=2)

        self.update_range_button = tk.Button(self.range_frame, text="Update Period and Frequency Range", command=self.update_range)
        self.update_range_button.grid(row=0, column=8, padx=2, pady=2)

        self.canvas_frame = tk.Frame(window)
        self.canvas_frame.grid(row=2, column=0, sticky="nsew")
        
        window.rowconfigure(2, weight=1)
        window.columnconfigure(0, weight=1)
        
        self.selected_points = []
        self.windows = []
        self.array = None
        self.selecting_window = False
        self.selected_window_edge = None
        self.editing_window = False
        self.press = None
        self.window_lines = []
        self.window_spans = []
        self.edit_motion_cid = None
        
        self.cursor_info_label = tk.Label(window, text="Cursor Position: (x, y)", anchor="w")
        self.cursor_info_label.grid(row=3, column=0, sticky='ew')
        
        self.fig, (self.ax, self.ax_fft, self.ax_fft_freq) = plt.subplots(3, 1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.canvas_frame.rowconfigure(0, weight=1)
        self.canvas_frame.columnconfigure(0, weight=1)

        self.reconnect_default_events()

        window.bind("<d>", lambda event: self.toggle_delete_mode())
        window.bind("<s>", lambda event: self.stop_choosing())
        window.bind("<a>", lambda event: self.prepare_to_select_window())
        window.bind("<m>", lambda event: self.merge_window())
        window.bind("<o>", lambda event: self.open_array())
        window.bind("<r>", lambda event: self.redo_selection())
        window.bind("<u>", lambda event: self.update_fft())
        window.bind("<l>", lambda event: self.load_windows())
        window.bind("<e>", lambda event: self.toggle_edit_mode())
        window.bind("<plus>", lambda event: self.zoom_in())
        window.bind("<minus>", lambda event: self.zoom_out())
        window.bind("=", lambda event: self.zoom_in())      # Shift + = is +
        window.bind("-", lambda event: self.zoom_out())

    def update_cursor_info(self, event):
        if event.inaxes is not None:
            x, y = event.xdata, event.ydata
            self.cursor_info_label.config(text=f"Cursor Position: (x={x:.2f}, y={y:.2f})")
        else:
            self.cursor_info_label.config(text="Cursor Position: (x, y)")

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.window.quit()
            self.window.destroy()
            sys.exit(0)

    def merge_window(self):
        if not self.windows:
            return

        self.windows.sort(key=lambda x: x[0])
        merged_windows = [self.windows[0]]
        for current_start, current_end in self.windows[1:]:
            last_merged_start, last_merged_end = merged_windows[-1]
            if current_start <= last_merged_end:
                merged_windows[-1] = (last_merged_start, max(last_merged_end, current_end))
            else:
                merged_windows.append((current_start, current_end))
        self.windows = merged_windows
        self.plot_windows()
        self.canvas.draw_idle()

    def toggle_delete_mode(self):
        self.deleting_window = not self.deleting_window
        if self.deleting_window:
            self.delete_window_button.configure(bg='red')
            self.canvas.mpl_connect('button_press_event', self.on_double_click)
        else:
            self.delete_window_button.configure(bg='gray')
            self.canvas.mpl_disconnect('button_press_event')
            self.reconnect_default_events()

    def toggle_edit_mode(self):
        self.editing_window = not self.editing_window
        if self.editing_window:
            self.edit_window_button.configure(bg='green')
        else:
            self.edit_window_button.configure(bg='gray')
        if self.editing_window:
            self.prepare_to_edit_window()
        else:
            self.stop_editing_window()
        self.selected_window_edge = None

    def prepare_to_edit_window(self):
        self.canvas.mpl_disconnect('button_press_event')
        self.canvas.mpl_connect('button_press_event', self.select_window_edge)
        self.edit_motion_cid = self.canvas.mpl_connect('motion_notify_event', self.update_window_edge)

    def on_double_click(self, event):
        if not self.deleting_window or event.inaxes != self.ax:
            return
        click_x = event.xdata
        closest_window = None
        min_distance = float('inf')
        for i, window in enumerate(self.windows):
            center = (window[0] + window[1]) / 2
            distance = abs(click_x - center)
            if distance < min_distance:
                min_distance = distance
                closest_window = i
        if closest_window is not None:
            self.windows.pop(closest_window)
            self.plot_windows()
            self.canvas.draw_idle()
            self.deleting_window = False
            self.delete_window_button.configure(bg='gray')

    def stop_editing_window(self):
        self.editing_window = False
        self.canvas.mpl_disconnect('motion_notify_event')
        self.canvas.mpl_disconnect('button_press_event')
        self.reconnect_default_events()
        self.selected_window_edge = None
        self.plot_windows()

    def load_windows(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.windows = []
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    self.windows.append(list(map(obspy.UTCDateTime, row)))
            self.plot_array(self.array)
            for window in self.windows:
                window[0] = int((float(window[0]) - self.starttime.timestamp)*self.sampling_rate)
                window[1] = int((float(window[1]) - self.starttime.timestamp)*self.sampling_rate)
                self.window_lines.append(self.ax.axvline(x=window[0], color='r', linestyle='--'))
                self.window_lines.append(self.ax.axvline(x=window[1], color='r', linestyle='--'))
                self.window_spans.append(self.ax.axvspan(window[0], window[1], alpha=0.2, color='r'))
            self.canvas.draw()

    def open_array(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.stream = obspy.read(file_path)
            self.array = self.stream[0].data[:]
            self.starttime = self.stream[0].stats.starttime
            self.sampling_rate = self.stream[0].stats.sampling_rate
            self.array = self.array - np.mean(self.array)
            self.plot_array(self.array)
            self.opened_file_base_name = os.path.splitext(os.path.basename(file_path))[0]
            self.windows = []

    def plot_array(self, array):
        self.ax.clear()
        self.ax.plot(array)
        if self.canvas is None:
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
            self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
            self.canvas.mpl_connect("motion_notify_event", self.update_cursor_info)
            self.reconnect_default_events()
        self.canvas.draw()

    def reconnect_default_events(self):
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)

    def prepare_to_select_window(self):
        self.selected_points = []
        self.selecting_window = True
        self.canvas.mpl_disconnect('button_press_event')
        self.canvas.mpl_disconnect('button_release_event')
        self.canvas.mpl_disconnect('motion_notify_event')
        self.canvas.mpl_disconnect("scroll_event")
        self.canvas.mpl_connect('button_press_event', self.select_points)

    def select_points(self, event):
        if not self.selecting_window or event.inaxes != self.ax or self.array is None:
            return
        idx = int(event.xdata)
        self.selected_points.append(idx)
        self.window_lines.append(self.ax.axvline(x=idx, color='r', linestyle='--'))
        if len(self.selected_points) == 2:
            self.windows.append(list(self.selected_points))
            self.window_spans.append(self.ax.axvspan(self.selected_points[0], self.selected_points[1], alpha=0.2, color='r'))
            self.selected_points = []
        self.canvas.draw()

    def stop_choosing(self):
        self.selecting_window = False
        self.canvas.mpl_disconnect('button_press_event')
        self.reconnect_default_events()

    def redo_selection(self):
        if self.windows:
            self.windows.pop()
            self.plot_array(self.array)
            for window in self.windows:
                self.ax.axvline(x=window[0], color='r', linestyle='--')
                self.ax.axvline(x=window[1], color='r', linestyle='--')
                self.ax.axvspan(window[0], window[1], alpha=0.2, color='r')
            self.canvas.draw()

    def save_windows(self):
        default_file_name = f"{self.opened_file_base_name}_selected_windows.csv" if self.opened_file_base_name else "default.csv"
        file_path = filedialog.asksaveasfilename(defaultextension='.csv', initialfile=default_file_name, filetypes=[("CSV files", "*.csv")])
        if file_path:
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                for window in self.windows:
                    new_window = [str(self.starttime + window[0]/self.sampling_rate),
                                  str(self.starttime + window[1]/self.sampling_rate)]
                    writer.writerow(new_window)

    def on_scroll(self, event):
        zoom_factor = 0.9 if event.button == 'up' else 1.1
        x_min, x_max = self.ax.get_xlim()
        try:
            y_min = np.min(self.array[int(x_min):int(x_max)])
        except:
            y_min = 0
        y_max = np.max(self.array[int(x_min):int(x_max)])
        new_width = (x_max - x_min) * zoom_factor
        new_height = (y_max - y_min)
        mid_x = (x_max + x_min) / 2
        mid_y = (y_max + y_min) / 2
        self.ax.set_xlim(mid_x - new_width / 2, mid_x + new_width / 2)
        self.ax.set_ylim(mid_y - new_height / 2, mid_y + new_height / 2)
        self.canvas.draw()

    def on_press(self, event):
        if event.inaxes != self.ax: 
            return
        self.press = (event.xdata, event.ydata)

    def on_motion(self, event):
        if self.editing_window:
            return
        if self.press is None: 
            return
        if event.inaxes != self.ax: 
            return
        dx = self.press[0] - event.xdata
        dy = self.press[1] - event.ydata
        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()
        self.ax.set_xlim(x_min + dx, x_max + dx)
        self.ax.set_ylim(y_min + dy, y_max + dy)
        self.canvas.draw()

    def on_release(self, event):
        if self.selected_window_edge is not None:
            self.canvas.mpl_disconnect('motion_notify_event')
            self.selected_window_edge = None
        self.press = None

    def on_resize(self, event):
        if self.canvas:
            self.canvas.get_tk_widget().config(width=event.width, height=event.height)

    def select_window_edge(self, event):
        if event.inaxes != self.ax or self.array is None:
            return
        click_x = event.xdata
        min_distance = float('inf')
        for i, window in enumerate(self.windows):
            for edge in window:
                distance = abs(click_x - edge)
                if distance < min_distance:
                    min_distance = distance
                    self.selected_window_edge = (i, window.index(edge))
        self.canvas.mpl_connect('motion_notify_event', self.update_window_edge)

    def update_window_edge(self, event):
        if not self.editing_window:
            return
        if self.selected_window_edge is None or event.inaxes != self.ax:
            return
        window_index, edge_index = self.selected_window_edge
        new_edge_position = int(event.xdata)
        self.windows[window_index][edge_index] = new_edge_position
        line_index = window_index * 2 + edge_index
        self.window_lines[line_index].set_xdata([new_edge_position, new_edge_position])
        self.window_spans[window_index].set_xy([[self.windows[window_index][0], 0],
                                                [self.windows[window_index][0], 1],
                                                [self.windows[window_index][1], 1],
                                                [self.windows[window_index][1], 0]])
        self.canvas.draw_idle()

    def minus_fft_calculator(self, sequence, windows, cosine_taper_length=50):
        taper = np.cos(np.linspace(0, np.pi, int(cosine_taper_length*2)))
        window_summation = np.zeros(len(sequence))
        for window in windows:
            start_dx = window[0]
            end_dx = window[1]
            temp_window = np.zeros(len(sequence))
            temp_window[start_dx:end_dx] = 1
            try:
                temp_window[start_dx-cosine_taper_length:start_dx] = temp_window[start_dx-cosine_taper_length:start_dx] * taper[0:cosine_taper_length]
                temp_window[end_dx:end_dx+cosine_taper_length] = temp_window[end_dx:end_dx+cosine_taper_length] * taper[cosine_taper_length:]
            except:
                pass
            window_summation = window_summation + temp_window

        sequence_windowed = sequence * window_summation
        sequence_windowed = sequence_windowed * scipy.signal.windows.hann(len(sequence))
        sequence_windowed_fft = np.fft.fft(sequence_windowed)
        sequence = sequence * scipy.signal.windows.hann(len(sequence))
        sequence_fft = np.fft.fft(sequence)
        minus_fft = sequence_fft - sequence_windowed_fft
        return minus_fft

    def update_fft(self):
        if self.array is None:
            return
        self.ax_fft.clear()
        minus_fft = self.minus_fft_calculator(self.array, self.windows, cosine_taper_length=50)
        fs = 6.6
        dt = 1/fs
        time_duration = len(self.array) * dt
        t = np.arange(0, time_duration, dt)
        freqs = np.fft.fftfreq(len(t), dt)
        period = 1 / freqs

        self.ax_fft.plot(period[1:len(period)//2], np.abs(minus_fft[1:len(minus_fft)//2])**2, linewidth=0.5, color='darkblue')
        self.ax_fft.set_xlabel('Period (s)')
        self.ax_fft.set_ylabel('Power')
        min_period = self.min_period
        max_period = self.max_period
        self.ax_fft.set_xlim(min_period, max_period)
        min_period_index = np.where(period[1:len(period)//2] < max_period)[0][0]
        max_period_index = np.where(period[1:len(period)//2] < min_period)[0][0]
        max_amp = np.max(np.abs(minus_fft[min_period_index+1:max_period_index+1])**2)
        self.ax_fft.set_ylim(0, max_amp*1.2)
        self.ax_fft.legend()

        self.ax_fft_freq.clear()
        self.ax_fft_freq.plot(freqs[1:len(freqs)//2], np.abs(minus_fft[1:len(minus_fft)//2])**2, linewidth=0.5, color='darkblue')
        self.ax_fft_freq.set_xlabel('Frequency (Hz)')
        self.ax_fft_freq.set_ylabel('Power')
        min_freq = self.min_freq
        max_freq = self.max_freq
        min_freq_index = np.where(freqs[1:len(freqs)//2] > min_freq)[0][0]
        max_freq_index = np.where(freqs[1:len(freqs)//2] > max_freq)[0][0]
        max_amp = np.max(np.abs(minus_fft[min_freq_index+1:max_freq_index+1])**2)
        self.ax_fft_freq.set_xlim(min_freq, max_freq)
        self.ax_fft_freq.set_ylim(0, max_amp*1.2)
        self.canvas.draw()

    def update_range(self):
        try:
            self.min_freq = float(self.min_freq_entry.get())
            self.max_freq = float(self.max_freq_entry.get())
            self.min_period = float(self.min_period_entry.get())
            self.max_period = float(self.max_period_entry.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter valid numeric values.")
            return
        self.update_fft()

    def plot_windows(self):
        for line in self.window_lines:
            line.remove()
        for span in self.window_spans:
            span.remove()
        self.window_lines.clear()
        self.window_spans.clear()
        for window in self.windows:
            line1 = self.ax.axvline(x=window[0], color='r', linestyle='--')
            line2 = self.ax.axvline(x=window[1], color='r', linestyle='--')
            span = self.ax.axvspan(window[0], window[1], alpha=0.2, color='r')
            self.window_lines.extend([line1, line2])
            self.window_spans.append(span)
        self.canvas.draw_idle()

    def zoom_in(self):
        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()
        zoom_factor = 0.9
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        new_x_range = (x_max - x_min) * zoom_factor
        new_y_range = (y_max - y_min) * zoom_factor
        self.ax.set_xlim(x_center - new_x_range / 2, x_center + new_x_range / 2)
        self.ax.set_ylim(y_center - new_y_range / 2, y_center + new_y_range / 2)
        self.canvas.draw()

    def zoom_out(self):
        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()
        zoom_factor = 1.1
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        new_x_range = (x_max - x_min) * zoom_factor
        new_y_range = (y_max - y_min) * zoom_factor
        self.ax.set_xlim(x_center - new_x_range / 2, x_center + new_x_range / 2)
        self.ax.set_ylim(y_center - new_y_range / 2, y_center + new_y_range / 2)
        self.canvas.draw()

window = tk.Tk()
app = ArrayPlotterApp(window)
window.mainloop()