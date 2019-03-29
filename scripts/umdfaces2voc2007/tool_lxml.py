#!/usr/bin/env python

from lxml import etree  
#import xml.etree.ElementTree as etree  
  
  
def createXML(trans,store_path):      
    annotation = etree.Element("annotation")  

    folder = etree.SubElement(annotation, "folder" )  
    folder.text = trans['folder']
      
    filename = etree.SubElement(annotation, "filename")   
    filename.text = trans['filename']

    '''path = etree.SubElement(annotation, "path")   
    path.text = "TEXT"'''

    source = etree.SubElement(annotation, "source")   
    source.text = "Unknown"

    owner = etree.SubElement(annotation, "owner")   
    flickrid = etree.SubElement(owner, "flickrid")   
    flickrid.text = "NULL"
    name = etree.SubElement(owner, "name")   
    name.text = "luuuyi"

    size = etree.SubElement(annotation, "size")   
    width = etree.SubElement(size, "width")   
    width.text = trans['width']
    height = etree.SubElement(size, "height")   
    height.text = trans['height']
    depth = etree.SubElement(size, "depth")   
    depth.text = trans['depth']

    segmented = etree.SubElement(annotation, "segmented")   
    segmented.text = "0"

    _object = etree.SubElement(annotation, "object")   
    name = etree.SubElement(_object, "name")   
    name.text = "face"
    pose = etree.SubElement(_object, "pose")   
    pose.text = "Unspecified"
    truncated = etree.SubElement(_object, "truncated")   
    truncated.text = "0"
    difficult = etree.SubElement(_object, "difficult")   
    difficult.text = "0"
    bndbox = etree.SubElement(_object, "bndbox")
    xmin = etree.SubElement(bndbox, "xmin")   
    xmin.text = trans['xmin']
    ymin = etree.SubElement(bndbox, "ymin")   
    ymin.text = trans['ymin']
    xmax = etree.SubElement(bndbox, "xmax")   
    xmax.text = trans['xmax']
    ymax = etree.SubElement(bndbox, "ymax")   
    ymax.text = trans['ymax'] 
       
    #print etree.tostring(annotation, pretty_print=True)   
      
    # write to file:    
    tree = etree.ElementTree(annotation) 
    file_name_list = trans['filename'].split('.')
    file_name = file_name_list[len(file_name_list)-2]
    tree.write(store_path+file_name+'.xml', pretty_print=True, xml_declaration=True, encoding='utf-8')