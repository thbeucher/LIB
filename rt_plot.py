import pyqtgraph as pg

class RTplot:
  '''
  This class alow to plot in real time
  pyQtGraph allow to plot in real time not by update the figure
  but by quickly clear and plot again
  '''
  def __init__(self, xName, yName):
    '''
    xName - string - abssice name
    yName - string - ordinate name
    '''
    self.pw = pg.plot()
    self.pw.setLabels(left=(yName))
    self.pw.setLabels(bottom=(xName))
    self.x = []
    self.y = []

  def updatePlot(self, newX, newY):
    '''
    Updates the figure
    newX - number - new abssice point
    newY - number - new ordinate point
    '''
    self.x.append(newX)
    self.y.append(newY)
    self.pw.plot(self.x, self.y, clear=True)
    pg.QtGui.QApplication.processEvents()

  def plotFromFile(self):
    '''
    Plots the mean cumulative rewards from file
    '''
    x, y = self.readXY()
    self.pw.plot(x, y, clear=True)
    pg.QtGui.QApplication.processEvents()

  def readXY(self):
    x, y = [], []
    with open('accuracy.txt', 'r') as f:
      tmp = f.readlines()
      for el in tmp:
        el = el.rstrip('\n')
        x.append(int(el.split('-')[0].split(':')[1]))
        y.append(float(el.split('-')[1].split(':')[1]))
    return x, y
