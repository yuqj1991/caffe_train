@echo off
"caffe/caffe.exe" train --solver=solver-12.prototxt --weights=det1.caffemodel
pause