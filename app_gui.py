# gui_minima.py
import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox,
    QPushButton, QTableWidget, QTableWidgetItem, QTextEdit, QMessageBox
)
from PyQt5.QtCore import Qt

# importa tu clase (debe estar en sistema_lineal.py en la misma carpeta)
from sistema_lineal import SistemaLineal


class Ventana(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gauss-Jordan")
        self.resize(720, 520)
        self._ui()

    def _ui(self):
        layout = QVBoxLayout(self)

        # ===== Controles superiores: SOLO m y n =====
        fila = QHBoxLayout()
        fila.addWidget(QLabel("Filas (m):"))
        self.sp_m = QSpinBox()
        self.sp_m.setRange(1, 30)
        self.sp_m.setValue(3)
        fila.addWidget(self.sp_m)

        fila.addSpacing(16)
        fila.addWidget(QLabel("Columnas (n):"))
        self.sp_n = QSpinBox()
        self.sp_n.setRange(1, 30)
        self.sp_n.setValue(3)
        fila.addWidget(self.sp_n)

        self.btn_crear = QPushButton("Crear matriz aumentada")
        self.btn_crear.clicked.connect(self.crear_matriz)
        fila.addStretch(1)
        fila.addWidget(self.btn_crear)

        layout.addLayout(fila)

        # ===== Tabla m x (n+1) =====
        self.tbl = QTableWidget(3, 4, self)  # por defecto 3x(3+1)
        self._config_headers()
        layout.addWidget(self.tbl)

        # ===== Acciones =====
        acciones = QHBoxLayout()
        self.btn_resolver = QPushButton("Resolver")
        self.btn_resolver.clicked.connect(self.resolver)
        acciones.addStretch(1)
        acciones.addWidget(self.btn_resolver)
        layout.addLayout(acciones)

        # ===== Salida =====
        self.txt_out = QTextEdit(self)
        self.txt_out.setReadOnly(True)
        self.txt_out.setPlaceholderText(
            "Aquí aparecerán TODOS los pasos del método y la interpretación final…"
        )
        layout.addWidget(self.txt_out)

        # ejemplo rápido
        ejemplo = [
            [1, 1, 1, 6],
            [2, -1, 1, 3],
            [-1, 2, 2, 14]
        ]
        self._llenar(ejemplo)

    def _config_headers(self):
        cols = self.tbl.columnCount()
        # n = cols-1 variables, última columna es b
        headers = [f"x{j+1}" for j in range(cols - 1)] + ["b"]
        self.tbl.setHorizontalHeaderLabels(headers)
        self.tbl.horizontalHeader().setStretchLastSection(True)

    def crear_matriz(self):
        m = self.sp_m.value()
        n = self.sp_n.value()
        self.tbl.setRowCount(m)
        self.tbl.setColumnCount(n + 1)
        self._config_headers()
        # limpiar celdas
        for i in range(m):
            for j in range(n + 1):
                self.tbl.setItem(i, j, QTableWidgetItem(""))

    def _llenar(self, mat):
        self.tbl.setRowCount(len(mat))
        self.tbl.setColumnCount(len(mat[0]))
        self._config_headers()
        for i, fila in enumerate(mat):
            for j, val in enumerate(fila):
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignCenter)
                self.tbl.setItem(i, j, item)

    def _leer_matriz(self):
        m = self.tbl.rowCount()
        n1 = self.tbl.columnCount()  # n + 1
        if m == 0 or n1 == 0:
            raise ValueError("La tabla está vacía.")
        M = []
        for i in range(m):
            fila = []
            for j in range(n1):
                it = self.tbl.item(i, j)
                txt = it.text().strip() if it else ""
                if txt == "":
                    txt = "0"  # mínima fricción: vacíos a 0
                try:
                    val = float(txt.replace(",", "."))
                except ValueError:
                    raise ValueError(f"Valor no numérico en ({i+1}, {j+1}): «{txt}».")
                fila.append(val)
            M.append(fila)
        return M

    def resolver(self):
        try:
            M = self._leer_matriz()
            sistema = SistemaLineal(M)
            # ⬇️ Mostrar TODA la traza + interpretación (tal cual devuelve tu clase)
            salida_completa = sistema.eliminacion_gaussiana()
            self.txt_out.setPlainText(salida_completa)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


def main():
    app = QApplication(sys.argv)
    win = Ventana()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
