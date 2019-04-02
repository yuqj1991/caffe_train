@echo off
"caffe/caffe.exe" train --solver=solver-24.prototxt --weights=det2.caffemodel
pause