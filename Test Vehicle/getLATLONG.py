from PyQt5 import QtCore, QtGui, QtWidgets,QtWebEngineWidgets,QtWebChannel
import json
import polyline
import googlemaps
import math
import matplotlib.pyplot as plt
import pymap3d as pm
gmaps = googlemaps.Client(key='AIzaSyDvBnnlw3QD0ai6GjdlqGBL5igbipfU0eA')
import csv
file_path = '/media/brain/Data/AV/GPSwaypoints.csv'
class Backend(QtCore.QObject):
    valueChanged = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._value = ""
        self.lat = 33.623047132268795
        self.long = 72.95787157924994

    @QtCore.pyqtProperty(str)
    def value(self):
        return self._value

    @QtCore.pyqtSlot(str, result=str)
    def getData(self,fromJS):
        #print(fromJS)
        #'33.62345578903128
        x = json.loads('{"lat":'+str(self.lat)+',"long":'+str(self.long)+'}')
        print(x)
        #print("function called from js")
        return json.dumps(x)

    @value.setter
    def value(self, v):
        self._value = v
        self.valueChanged.emit(v)

    def set_lat_long(self, lat,long):
        self.lat = lat
        self.long = long
        print("in set ftn",self.lat,self.long)
class Widget(QtWidgets.QWidget):
    x_coord = 0.0
    y_coord = 0.0
    z_coord = 0.0
    def __init__(self, parent=None):
        self.backend = ""
        super().__init__(parent)
        self.webEngineView = QtWebEngineWidgets.QWebEngineView()
        self.label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        #self.showFullScreen()
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.webEngineView, stretch=1)
        #lay.addWidget(self.label, stretch=1)
        self.backend = Backend(self)
        #backend.valueChanged.connect(self.label.setText)
        self.backend.valueChanged.connect(self.foo_function)
        self.channel = QtWebChannel.QWebChannel()
        self.channel.registerObject("backend", self.backend)
        self.webEngineView.page().setWebChannel(self.channel)
        
        path = "/media/brain/Data/AV/Maps/html/index.html"
        self.webEngineView.setUrl(QtCore.QUrl.fromLocalFile(path))
    def getElevation(self, value): 
        raw_elevation = gmaps.elevation(locations=value)
        strele = json.dumps(raw_elevation)
        upele = json.loads(strele)
        final = []
        for i in range(len(value)):
            final.append(upele[i]['elevation'])
        #minimum = min(final)
        #ret = [x - minimum for x in final]
        return final
    def getENU(self, lat, longi, alt):
        x = []
        y = []
        z = []
        for i in range(len(lat)):
            ENU = pm.geodetic2enu(lat[i], longi[i], alt[i], lat[0], longi[0], alt[0])
            x.append(ENU[0])
            y.append(ENU[1])
            z.append(ENU[2])
        return x,y,z
    #def getECEF(self, lat, lon, alt):
    #    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    #    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    #    x, y, z = pyproj.transform(lla, ecef, lon, lat, alt, radians=False)
    #    return x, y, z
    def getAngle(self,x,y,z):
        theeta = []
        for i in range(len(x)):
            if(i==0):
                continue
            num = math.sqrt(x[i]**2 + y[i]**2)
            angle = math.atan2(num,z[i])
            theeta.append(90 - math.degrees(angle))
        return theeta
    
    @QtCore.pyqtSlot(str)
    def foo_function(self, value):
        print("Path Initialized")
        #lat = value.split(':')[1].split(',')[0].strip()
        #long = value.split(':')[2].split('\n')[0].strip()
        #dest = lat + ',' + long
        #print("Destination set as(Lat,Long):",dest)
        #print("\n",value)
        #waypoints = gmaps.directions("33.62336044507634,72.9550766068185",
        #                             dest
        #                            )
        #strway = json.dumps(value)
        #self.backend.getData(False)
        PrettyWayPoints = json.loads(value)
        polylines_enc = PrettyWayPoints['routes'][0]['overview_polyline']
        rawLL = polyline.decode(polylines_enc)
        elevation = self.getElevation(rawLL)
        lati,longi=zip(*rawLL)
        #_____________________________________________________________
        zippedPoints = [str(i)+","+str(j)+"|" for i,j in rawLL]
        zippedStr = ''.join(zippedPoints)
        zippedStr = zippedStr[:-1]

        snaps = gmaps.nearest_roads(zippedStr)

        jsonSnaps = json.loads(json.dumps(snaps))
        
        roadsLati=[]
        roadsLongi=[]
        for i in jsonSnaps:
            roadsLati.append(i['location']['latitude'])
            roadsLongi.append(i['location']['longitude'])

        #____________________________________________________________
        self.x_coord,self.y_coord,self.z_coord = self.getENU(lati,longi,elevation)
        theeta = self.getAngle(self.x_coord,self.y_coord,self.z_coord)
        #difAngle = [y-x for x, y in zip(theeta[:-1], theeta[1:])]
        
        print(max(theeta))
        ax = plt.axes(projection="3d")
        plt.cla()
        #plt.scatter(x,y)
        #plt.scatter(roadsLongi,roadsLati,marker='x')
        ax.plot3D(self.x_coord,self.y_coord,self.z_coord)
        #for i in range(len(self.x_coord)):
        with open(file_path, 'w') as file:
            writer = csv.writer(file)
            for i in range(len(self.x_coord)):
                writer.writerow([self.x_coord[i],self.y_coord[i],self.z_coord[i]])
            
        plt.pause(0.05)

#if __name__ == "__main__":
#    import sys
#    app = QtWidgets.QApplication(sys.argv)
#    w = Widget()
#    w.show()
#    sys.exit(app.exec_())

