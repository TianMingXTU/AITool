import sys
import threading
import asyncio
import speech_recognition as sr
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from PyQt5.QtWidgets import (
    QApplication,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QLabel,
    QComboBox,
    QHBoxLayout,
    QTextEdit,
    QProgressBar,
    QFrame,
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QColor, QFont
import pyaudio
from zhipuai import ZhipuAI
from datetime import datetime
import logging
from PyQt5.QtCore import QRunnable

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# ChatGLM integration
client = ZhipuAI(
    api_key="8013501cb9918c19098c4fc2372f0ba8.neocjMKUYjhDBNbi"
)  # Fill in your API key

# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024  # Size of the audio chunk for each read

# Global messages list to store conversation history for context
messages = []


class AudioRecorder:
    """Class responsible for recording audio in chunks and passing data for visualization."""

    def __init__(self, waveform_plot_widget, spectrum_plot_widget):
        self.waveform_plot_widget = waveform_plot_widget
        self.spectrum_plot_widget = spectrum_plot_widget
        self.pyaudio_instance = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self.timer = QtCore.QTimer()
        logging.info("AudioRecorder initialized.")

    def start_recording(self):
        """Start the audio recording."""
        if self.is_recording:
            logging.warning("Recording already in progress.")
            return

        self.stream = self.pyaudio_instance.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )
        self.is_recording = True
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(50)  # Update the plot every 50ms
        logging.info("Audio recording started.")

    def stop_recording(self):
        """Stop the audio recording."""
        if not self.is_recording:
            logging.warning("Recording is not active.")
            return

        self.is_recording = False
        self.timer.stop()
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.pyaudio_instance.terminate()
        logging.info("Audio recording stopped.")

    def update_plots(self):
        """Read audio data from the stream and update both waveform and spectrum plots."""
        if self.is_recording:
            try:
                data = self.stream.read(CHUNK, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)

                # Update the waveform plot
                self.waveform_plot_widget.plot_waveform(audio_data)

                # Update the spectrum plot (FFT)
                fft_data = np.abs(np.fft.fft(audio_data))[: CHUNK // 2]
                self.spectrum_plot_widget.plot_spectrum(fft_data)
            except Exception as e:
                logging.error(f"Error updating audio plots: {e}")


class AudioPlotWidget(pg.PlotWidget):
    """Widget for displaying real-time audio waveform."""

    def __init__(self):
        super().__init__()
        self.setYRange(-32768, 32767)  # For int16 audio data
        self.setXRange(0, CHUNK)
        self.waveform_curve = self.plot(pen=pg.mkPen("g", width=2))

    def plot_waveform(self, audio_data):
        """Plot the audio waveform in real-time."""
        self.waveform_curve.setData(audio_data)


class SpectrumPlotWidget(pg.PlotWidget):
    """Widget for displaying real-time audio spectrum with gradient color."""

    def __init__(self):
        super().__init__()
        self.setYRange(0, 1000)  # Adjust based on typical FFT amplitude
        self.setXRange(0, CHUNK // 2)  # Half-spectrum (positive frequencies)
        self.spectrum_curve = self.plot(pen=pg.mkPen(color=(0, 255, 0), width=2))
        self.gradient_color = [
            (0, QColor(0, 255, 0)),
            (0.5, QColor(255, 255, 0)),
            (1, QColor(255, 0, 0)),
        ]

    def plot_spectrum(self, fft_data):
        """Plot the FFT spectrum in real-time with gradient color."""
        color_map = pg.ColorMap(
            [0, 0.5, 1], [color[1] for color in self.gradient_color]
        )
        gradient = color_map.getLookupTable(0.0, 1.0, len(fft_data))
        pen = pg.mkPen(color=pg.mkColor(gradient[int(len(fft_data) / 2)]))
        self.spectrum_curve.setPen(pen)
        self.spectrum_curve.setData(fft_data)


class SpeechRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.is_listening = False
        self.selected_language = "en-US"
        self.selected_theme = "Hacker Style"  # Default theme
        self.initUI()
        self.audio_recorder = AudioRecorder(
            self.audio_plot_widget, self.spectrum_plot_widget
        )

    def initUI(self):
        self.setWindowTitle("ChatGLM Voice Assistant")
        self.setGeometry(300, 300, 900, 800)
        self.setAutoFillBackground(True)
        self.apply_theme(self.selected_theme)

        layout = QVBoxLayout()

        self.text_area = QTextEdit(self)
        self.text_area.setReadOnly(True)
        self.text_area.setFixedHeight(350)
        layout.addWidget(self.text_area)

        self.response_area = QTextEdit(self)
        self.response_area.setReadOnly(True)
        self.response_area.setFixedHeight(300)
        layout.addWidget(self.response_area)

        self.status_label = QLabel("Click 'Start Listening' to begin", self)
        layout.addWidget(self.status_label)

        self.theme_dropdown = QComboBox(self)
        self.theme_dropdown.addItems(
            ["Hacker Style", "Modern Style", "Light Style", "Dark Style"]
        )
        self.theme_dropdown.currentIndexChanged.connect(self.change_theme)
        layout.addWidget(QLabel("Select Theme:"))
        layout.addWidget(self.theme_dropdown)

        self.language_dropdown = QComboBox(self)
        self.language_dropdown.addItems(
            [
                "English (US)",
                "English (UK)",
                "Chinese (Simplified)",
                "Spanish",
                "French",
                "German",
            ]
        )
        self.language_dropdown.currentIndexChanged.connect(self.change_language)
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Language: "))
        h_layout.addWidget(self.language_dropdown)
        layout.addLayout(h_layout)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.button = QPushButton("Start Listening", self)
        self.button.clicked.connect(self.toggle_listening)
        layout.addWidget(self.button)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)

        self.audio_plot_widget = AudioPlotWidget()
        self.audio_plot_widget.setFixedHeight(100)
        layout.addWidget(self.audio_plot_widget)

        self.spectrum_plot_widget = SpectrumPlotWidget()
        self.spectrum_plot_widget.setFixedHeight(100)
        layout.addWidget(self.spectrum_plot_widget)

        self.setLayout(layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_progress_bar)

    def apply_theme(self, theme):
        """Apply selected theme to the application."""
        if theme == "Hacker Style":
            self.setStyleSheet(
                """
                QWidget {
                    background-color: #000000;
                    color: #00FF00;
                }
                QPushButton {
                    background-color: #00FF00;
                    color: #000000;
                    font-weight: bold;
                    border-radius: 10px;
                }
                QPushButton:hover {
                    background-color: #000000;
                    color: #00FF00;
                }
                QLabel, QComboBox {
                    color: #00FF00;
                }
                QTextEdit {
                    background-color: #1E1E1E;
                    color: #00FF00;
                    font-family: 'Courier New';
                    border: 2px solid #00FF00;
                }
                QProgressBar {
                    background-color: #1E1E1E;
                    color: #00FF00;
                    text-align: center;
                }
            """
            )
        elif theme == "Modern Style":
            self.setStyleSheet(
                """
                QWidget {
                    background-color: #F0F0F0;
                    color: #000000;
                }
                QPushButton {
                    background-color: #007BFF;
                    color: #FFFFFF;
                    font-weight: bold;
                    border-radius: 10px;
                }
                QPushButton:hover {
                    background-color: #0056b3;
                }
                QLabel, QComboBox {
                    color: #000000;
                }
                QTextEdit {
                    background-color: #FFFFFF;
                    color: #000000;
                    border: 1px solid #007BFF;
                }
                QProgressBar {
                    background-color: #E0E0E0;
                    color: #007BFF;
                    text-align: center;
                }
            """
            )
        elif theme == "Light Style":
            self.setStyleSheet(
                """
                QWidget {
                    background-color: #FFFFFF;
                    color: #000000;
                }
                QPushButton {
                    background-color: #F0F0F0;
                    color: #000000;
                    border-radius: 10px;
                }
                QPushButton:hover {
                    background-color: #E0E0E0;
                }
                QLabel, QComboBox {
                    color: #000000;
                }
                QTextEdit {
                    background-color: #FFFFFF;
                    color: #000000;
                    border: 1px solid #000000;
                }
                QProgressBar {
                    background-color: #F0F0F0;
                    color: #000000;
                    text-align: center;
                }
            """
            )
        elif theme == "Dark Style":
            self.setStyleSheet(
                """
                QWidget {
                    background-color: #2E2E2E;
                    color: #FFFFFF;
                }
                QPushButton {
                    background-color: #555555;
                    color: #FFFFFF;
                    border-radius: 10px;
                }
                QPushButton:hover {
                    background-color: #444444;
                }
                QLabel, QComboBox {
                    color: #FFFFFF;
                }
                QTextEdit {
                    background-color: #1E1E1E;
                    color: #FFFFFF;
                    border: 1px solid #555555;
                }
                QProgressBar {
                    background-color: #444444;
                    color: #FFFFFF;
                    text-align: center;
                }
            """
            )

    def change_theme(self):
        """Change theme based on the selection in the dropdown."""
        selected_theme = self.theme_dropdown.currentText()
        self.apply_theme(selected_theme)
        logging.info(f"Theme changed to {selected_theme}")

    def change_language(self):
        """Change language based on the selection in the dropdown."""
        languages = {
            "English (US)": "en-US",
            "English (UK)": "en-GB",
            "Chinese (Simplified)": "zh-CN",
            "Spanish": "es-ES",
            "French": "fr-FR",
            "German": "de-DE",
        }
        selected = self.language_dropdown.currentText()
        self.selected_language = languages.get(selected, "en-US")
        self.status_label.setText(f"Language changed to {selected}")
        logging.info(f"Language changed to {selected}")

    def toggle_listening(self):
        """Toggle between starting and stopping the listening process."""
        if not self.is_listening:
            self.is_listening = True
            self.button.setText("Stop Listening")
            self.status_label.setText("Listening...")
            self.text_area.append(f"Listening... [{self.get_current_time()}]\n")
            self.text_area.moveCursor(QtGui.QTextCursor.End)
            self.start_listening_thread()
            self.progress_bar.setValue(50)
            self.timer.start(100)
            self.audio_recorder.start_recording()
            logging.info("Started listening.")
        else:
            self.is_listening = False
            self.button.setText("Start Listening")
            self.status_label.setText("Stopped listening.")
            self.text_area.append(f"Stopped listening. [{self.get_current_time()}]\n")
            self.text_area.moveCursor(QtGui.QTextCursor.End)
            self.progress_bar.setValue(0)
            self.timer.stop()
            self.audio_recorder.stop_recording()
            logging.info("Stopped listening.")

    def start_listening_thread(self):
        """Start a new thread for the listening task."""
        worker = ListenWorker(
            self.recognizer, self.microphone, self.selected_language, self, self
        )
        threading.Thread(target=worker.run, daemon=True).start()

    def update_progress_bar(self):
        """Update the progress bar dynamically."""
        current_value = self.progress_bar.value()
        if current_value >= 100:
            self.progress_bar.setValue(0)
        else:
            self.progress_bar.setValue(current_value + 5)

    def update_response_area(self, response):
        """Update the response area with ChatGLM model's reply."""
        self.response_area.append(f"ChatGLM [{self.get_current_time()}]: {response}\n")
        self.response_area.moveCursor(QtGui.QTextCursor.End)

    @staticmethod
    def get_current_time():
        """Return the current time formatted as HH:MM:SS."""
        return datetime.now().strftime("%H:%M:%S")


class ListenWorker(QRunnable):
    def __init__(self, recognizer, microphone, language, parent, ui):
        super().__init__()
        self.recognizer = recognizer
        self.microphone = microphone
        self.language = language
        self.parent = parent
        self.ui = ui

    def run(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            while self.parent.is_listening:
                try:
                    audio = self.recognizer.listen(
                        source, timeout=10, phrase_time_limit=15
                    )
                    text = self.recognizer.recognize_google(
                        audio, language=self.language
                    )
                    self.parent.text_area.append(
                        f"Recognized [{self.ui.get_current_time()}]: {text}\n"
                    )
                    self.parent.text_area.moveCursor(QtGui.QTextCursor.End)
                    self.parent.status_label.setText("Recognized.")
                    messages.append({"role": "user", "content": text})
                    asyncio.run(self.send_to_chatglm(text))
                except sr.WaitTimeoutError:
                    self.parent.status_label.setText(
                        "Listening timed out, please try again."
                    )
                except sr.UnknownValueError:
                    logging.warning("Speech not understood.")
                except sr.RequestError as e:
                    self.parent.text_area.append(f"API request failed: {e}\n")
                    self.parent.text_area.moveCursor(QtGui.QTextCursor.End)
                    self.parent.status_label.setText("API request failed.")
                    logging.error(f"Speech recognition API request failed: {e}")
                except Exception as e:
                    self.parent.text_area.append(f"An error occurred: {e}\n")
                    self.parent.text_area.moveCursor(QtGui.QTextCursor.End)
                    self.parent.status_label.setText(f"An error occurred.")
                    logging.error(f"Error during listening: {e}")

    async def send_to_chatglm(self, user_input):
        """Send recognized text to ChatGLM model and display the response."""
        words = "你是一个专业的助手，请你简练的回答问题，不得超过200字。"
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="glm-4-0520",
            messages=messages,
        )
        reply = response.choices[0].message.content
        messages.append({"role": "assistant", "content": words + reply})
        self.ui.update_response_area(reply)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = SpeechRecognitionApp()
    ex.show()
    sys.exit(app.exec_())
