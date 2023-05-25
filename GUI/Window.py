from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import QPen, QBrush
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsView, QGraphicsItem


class DrawItem(QGraphicsItem):
    def __init__(self, brush, pen, rect):
        super().__init__()
        self.brush = brush
        self.pen = pen
        self.rect = rect

    def boundingRect(self):
        return self.rect

    def paint(self, painter, option, widget=None):
        painter.setBrush(self.brush)
        painter.setPen(self.pen)
        painter.drawRect(self.rect)


class PaintScene(QGraphicsScene):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pen = QPen(Qt.yellow, 2, Qt.SolidLine)
        self.brush = QBrush(Qt.SolidPattern)
        self.drawing = False
        self.lastPoint = None

    def mousePressEvent(self, event):
        self.drawing = True
        self.lastPoint = event.scenePos()

    def mouseMoveEvent(self, event):
        if self.drawing:
            currentPoint = event.scenePos()
            self.addItem(DrawItem(self.brush, self.pen,
                                  self.getRect(self.lastPoint, currentPoint)))
            self.lastPoint = currentPoint

    def mouseReleaseEvent(self, event):
        self.drawing = False

    def drawBackground(self, painter, rect):
        super().drawBackground(painter, rect)
        painter.setPen(self.pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(rect)

    def getRect(self, point1, point2):
        topLeft = QPointF(min(point1.x(), point2.x()), min(point1.y(), point2.y()))
        bottomRight = QPointF(max(point1.x(), point2.x()), max(point1.y(), point2.y()))
        return QRectF(topLeft, bottomRight)


class PaintWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(550, 250, 800, 600)
        self.setWindowTitle('Paint')

        self.scene = PaintScene(self)
        self.view = QGraphicsView(self.scene)
        self.setCentralWidget(self.view)


if __name__ == '__main__':
    app = QApplication([])
    window = PaintWindow()
    window.show()
    app.exec_()
