import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class DataSample:
    def __init__(self, data):
        self._data = data

    def mean(self):
        if not self._data: return 0
        return sum(self._data) / len(self._data)
    
    def std(self):
        if not self._data: return 0
        m = self.mean()
        return (sum((x-m)**2 for x in self._data) / len(self._data))**0.5
    
    def cv(self):
        m = self.mean()
        if m == 0: return 0
        return (self.std() / m) * 100

class KalmanFilter1D(DataSample):
    def __init__(self, data, q, r, initial_estimate=0, initial_error=1):
        super().__init__(data)
        self._q = q
        self._r = r
        self._estimate = initial_estimate
        self._error = initial_error
    
    def apply_filter(self):
        filtered_data = []
        for z in self._data:
            # Tahmin (Prediction)
            self._error = self._error + self._q
            
            # Güncelleme (Update)
            K = self._error / (self._error + self._r)
            self._estimate = self._estimate + K * (z - self._estimate)
            self._error = (1 - K) * self._error
            
            filtered_data.append(self._estimate)
            
        return filtered_data

class KalmanGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kalman Filter Signal Analyzer")
        self.resize(800, 600)

        layout = QVBoxLayout(self)

        self._input_field = QLineEdit()
        self._q_input = QLineEdit()
        self._r_input = QLineEdit()
        self._button = QPushButton("Analyze")
        self._result_label = QLabel("Statistical Result:")
        self._figure = Figure()
        self._canvas = FigureCanvas(self._figure)

        # Etiketler (QLabel) ve Kutular (QLineEdit) sırasıyla ekleniyor
        layout.addWidget(QLabel("Measurement Data (e.g. 10, 12, 14, 13, 16):"))
        layout.addWidget(self._input_field)
        
        layout.addWidget(QLabel("Process Noise Q (e.g. 0.1):"))
        layout.addWidget(self._q_input)
        
        layout.addWidget(QLabel("Measurement Noise R (e.g. 1.0):"))
        layout.addWidget(self._r_input)
        
        layout.addWidget(self._button)
        layout.addWidget(self._result_label)
        layout.addWidget(self._canvas)

        self._button.clicked.connect(self.analyze)

    def analyze(self):
        try:
            # Kutulardaki metinleri al ve sayıya çevir
            raw_data = [float(x.strip()) for x in self._input_field.text().split(',')]
            q_val = float(self._q_input.text())
            r_val = float(self._r_input.text())
        except ValueError:
            QMessageBox.warning(self, "Hata", "Lütfen kutuları boş bırakmayın ve geçerli sayılar girin!")
            return

        # Kalman Filtresini çalıştır
        kf = KalmanFilter1D(raw_data, q_val, r_val, initial_estimate=raw_data[0])
        filtered_data = kf.apply_filter()

        # İstatistikleri ekrana yazdır
        self._result_label.setText(
            f"Statistical Result:\n"
            f"Mean: {kf.mean():.2f} | Std: {kf.std():.2f} | CV: {kf.cv():.2f}%"
        )

        # Grafikleri çizdir
        self.draw_plots(raw_data, filtered_data)

    def draw_plots(self, raw, filtered):
        self._figure.clear()
        
        # 1. Çizgi Grafiği (Yan yana olması için 121)
        ax1 = self._figure.add_subplot(121)
        ax1.plot(raw, label="Raw Measurement", marker='o', linestyle='dashed')
        ax1.plot(filtered, label="Kalman Filtered", color='red', linewidth=2)
        ax1.set_title("Signal Tracking")
        ax1.legend()

        # 2. Histogram Grafiği (Yan yana olması için 122)
        ax2 = self._figure.add_subplot(122)
        ax2.hist(raw, alpha=0.5, label="Raw")
        ax2.hist(filtered, alpha=0.5, label="Filtered", color='red')
        ax2.set_title("Data Distribution")
        ax2.legend()

        self._canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = KalmanGUI()
    window.show()
    sys.exit(app.exec_())