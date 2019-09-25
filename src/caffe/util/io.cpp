#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <sstream>
#include <iostream>
#include <string>
#include <stdint.h>
#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <map>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

const int kProtoReadBytesLimit = INT_MAX;  // Max size of 2 GB minus 1 byte.

namespace caffe {

using namespace boost::property_tree;  // NOLINT(build/namespaces)
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;

bool ReadProtoFromTextFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  FileInputStream* input = new FileInputStream(fd);
  bool success = google::protobuf::TextFormat::Parse(input, proto);
  delete input;
  close(fd);
  return success;
}

void WriteProtoToTextFile(const Message& proto, const char* filename) {
  int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  FileOutputStream* output = new FileOutputStream(fd);
  CHECK(google::protobuf::TextFormat::Print(proto, output));
  delete output;
  close(fd);
}

bool ReadProtoFromBinaryFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  ZeroCopyInputStream* raw_input = new FileInputStream(fd);
  CodedInputStream* coded_input = new CodedInputStream(raw_input);
  coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);

  bool success = proto->ParseFromCodedStream(coded_input);

  delete coded_input;
  delete raw_input;
  close(fd);
  return success;
}

void WriteProtoToBinaryFile(const Message& proto, const char* filename) {
  fstream output(filename, ios::out | ios::trunc | ios::binary);
  CHECK(proto.SerializeToOstream(&output));
}

#ifdef USE_OPENCV
cv::Mat ReadImageToCVMat(const string& filename, const int height,
    const int width, const int min_dim, const int max_dim,
    const bool is_color) {
  cv::Mat cv_img;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
  if (!cv_img_origin.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return cv_img_origin;
  }
  if (min_dim > 0 || max_dim > 0) {
    int num_rows = cv_img_origin.rows;
    int num_cols = cv_img_origin.cols;
    int min_num = std::min(num_rows, num_cols);
    int max_num = std::max(num_rows, num_cols);
    float scale_factor = 1;
    if (min_dim > 0 && min_num < min_dim) {
      scale_factor = static_cast<float>(min_dim) / min_num;
    }
    if (max_dim > 0 && static_cast<int>(scale_factor * max_num) > max_dim) {
      // Make sure the maximum dimension is less than max_dim.
      scale_factor = static_cast<float>(max_dim) / max_num;
    }
    if (scale_factor == 1) {
      cv_img = cv_img_origin;
    } else {
      cv::resize(cv_img_origin, cv_img, cv::Size(0, 0),
                 scale_factor, scale_factor);
    }
  } else if (height > 0 && width > 0) {
    cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
  } else {
    cv_img = cv_img_origin;
  }
  return cv_img;
}

cv::Mat ReadImageToCVMat(const string& filename, const int height,
    const int width, const int min_dim, const int max_dim) {
  return ReadImageToCVMat(filename, height, width, min_dim, max_dim, true);
}

cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width, const bool is_color) {
  return ReadImageToCVMat(filename, height, width, 0, 0, is_color);
}

cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width) {
  return ReadImageToCVMat(filename, height, width, true);
}

cv::Mat ReadImageToCVMat(const string& filename,
    const bool is_color) {
  return ReadImageToCVMat(filename, 0, 0, is_color);
}

cv::Mat ReadImageToCVMat(const string& filename) {
  return ReadImageToCVMat(filename, 0, 0, true);
}

// Do the file extension and encoding match?
static bool matchExt(const std::string & fn,
                     std::string en) {
  size_t p = fn.rfind('.') + 1;
  std::string ext = p != fn.npos ? fn.substr(p) : fn;
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  std::transform(en.begin(), en.end(), en.begin(), ::tolower);
  if ( ext == en )
    return true;
  if ( en == "jpg" && ext == "jpeg" )
    return true;
  return false;
}

bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const int min_dim, const int max_dim,
    const bool is_color, const std::string & encoding, Datum* datum) {
  cv::Mat cv_img = ReadImageToCVMat(filename, height, width, min_dim, max_dim,
                                    is_color);
  if (cv_img.data) {
    if (encoding.size()) {
      if ( (cv_img.channels() == 3) == is_color && !height && !width &&
          !min_dim && !max_dim && matchExt(filename, encoding) ) {
        datum->set_channels(cv_img.channels());
        datum->set_height(cv_img.rows);
        datum->set_width(cv_img.cols);
        return ReadFileToDatum(filename, label, datum);
      }
      EncodeCVMatToDatum(cv_img, encoding, datum);
      datum->set_label(label);
      return true;
    }
    CVMatToDatum(cv_img, datum);
    datum->set_label(label);
    return true;
  } else {
    return false;
  }
}

void GetImageSize(const string& filename, int* height, int* width) {
  cv::Mat cv_img = cv::imread(filename);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return;
  }
  *height = cv_img.rows;
  *width = cv_img.cols;
}

bool ReadRichImageToAnnotatedDatum(const string& filename,
    const string& labelfile, const int height, const int width,
    const int min_dim, const int max_dim, const bool is_color,
    const string& encoding, const AnnotatedDatum_AnnotationType type,
    const AnnotatedDatum_AnnoataionAttriType attri_type,
    const string& labeltype, const std::map<string, int>& name_to_label,
    AnnotatedDatum* anno_datum) {
  // Read image to datum.
  bool status = ReadImageToDatum(filename, -1, height, width,
                                 min_dim, max_dim, is_color, encoding,
                                 anno_datum->mutable_datum());
  if (status == false) {
    return status;
  }
  anno_datum->clear_annotation_group();
  if (!boost::filesystem::exists(labelfile)) {
    return true;
  }
  // annno type bbox or attributes
  switch (type) {
    case AnnotatedDatum_AnnotationType_BBOX:
      int ori_height, ori_width;
      GetImageSize(filename, &ori_height, &ori_width);
      switch(attri_type){
        case AnnotatedDatum_AnnoataionAttriType_FACE:
        if (labeltype == "xml") {
          return ReadXMLToAnnotatedDatum(labelfile, ori_height, ori_width,
                                        name_to_label, anno_datum);
        } else if (labeltype == "json") {
          return ReadJSONToAnnotatedDatum(labelfile, ori_height, ori_width,
                                          name_to_label, anno_datum);
        } else if (labeltype == "txt") {
          return ReadTxtToAnnotatedDatum(labelfile, ori_height, ori_width,
                                        anno_datum);
        } else {
          LOG(FATAL) << "Unknown label file type.";
          return false;
        }
        break;
      case AnnotatedDatum_AnnoataionAttriType_NORMALL:
        if (labeltype == "txt") {
          return ReadTxtToAnnotatedDatum(labelfile, ori_height, ori_width,
                                        anno_datum);
        }else if(labeltype == "xml") {
          return ReadXMLToAnnotatedDatum(labelfile, ori_height, ori_width,
                                        name_to_label, anno_datum); 
          }else {
          LOG(FATAL) << "Unknown label file type.";
          return false;
        }
        break;
      default:
        LOG(FATAL) << "Unknown attri type.";
        return false;
      }
      break;
    default:
      LOG(FATAL) << "Unknown annotation type.";
      return false;
  }
}

bool ReadRichFaceToAnnotatedDatum(const string& filename,
    const string& labelfile, const int height, const int width,
    const int min_dim, const int max_dim, const bool is_color,
    const string& encoding, const AnnoFaceDatum_AnnotationType type,
    const string& labeltype, AnnoFaceDatum* anno_datum) {
  // Read image to datum.
  //return true;
  bool status = ReadImageToDatum(filename, -1, height, width,
                                 min_dim, max_dim, is_color, encoding,
                                 anno_datum->mutable_datum());
  if (status == false) {
    return status;
  }
  anno_datum->clear_annoface();
  if (!boost::filesystem::exists(labelfile)) {
    return true;
  }
  // annno type bbox or attributes
  switch (type) {
    case AnnoFaceDatum_AnnotationType_FACEMARK:
      int ori_height, ori_width;
      GetImageSize(filename, &ori_height, &ori_width);
      if (labeltype == "txt") {
        return ReadFaceAttriTxtToAnnotatedDatum(labelfile, ori_height, ori_width,
                                       anno_datum);
      } else {
        LOG(FATAL) << "Unknown label file type.";
        return false;
      }
      break;
    default:
      LOG(FATAL) << "Unknown annotation type.";
      return false;
  }
}

bool ReadRichFacePoseToAnnotatedDatum(const string& filename,
    const string& labelfile, const int height, const int width,
    const int min_dim, const int max_dim, const bool is_color,
    const string& encoding, const AnnoFaceAttributeDatum_AnnoType type,
    const string& labeltype, AnnoFaceAttributeDatum* anno_datum){
  // Read image to datum.
  bool status = ReadImageToDatum(filename, -1, height, width,
                                 min_dim, max_dim, is_color, encoding,
                                 anno_datum->mutable_datum());
  if (status == false) {
    return status;
  }
  anno_datum->clear_facepose();
  if (!boost::filesystem::exists(labelfile)) {
    return true;
  }
  // annno type bbox or attributes
  switch (type) {
    case AnnoFaceAttributeDatum_AnnoType_FACEPOSE:
      int ori_height, ori_width;
      GetImageSize(filename, &ori_height, &ori_width);
      if (labeltype == "txt") {
        return ReadumdfaceTxtToAnnotatedDatum(labelfile, ori_height, ori_width,
                                       anno_datum);
      } else {
        LOG(FATAL) << "Unknown label file type.";
        return false;
      }
      break;
    default:
      LOG(FATAL) << "Unknown annotation type.";
      return false;
  }
}

bool ReadRichFaceContourToAnnotatedDatum(const string& filename,
    const string& labelfile, const int height, const int width,
    const int min_dim, const int max_dim, const bool is_color,
    const string& encoding, const AnnoFaceContourDatum_AnnoType type,
    const string& labeltype, AnnoFaceContourDatum* anno_datum){
  // Read image to datum.
  bool status = ReadImageToDatum(filename, -1, height, width,
                                 min_dim, max_dim, is_color, encoding,
                                 anno_datum->mutable_datum());
  if (status == false) {
    return status;
  }
  anno_datum->clear_facecontour();
  if (!boost::filesystem::exists(labelfile)) {
    return true;
  }
  // annno type bbox or attributes
  switch (type) {
    case AnnoFaceContourDatum_AnnoType_FACECONTOUR:
      int ori_height, ori_width;
      GetImageSize(filename, &ori_height, &ori_width);
      if (labeltype == "txt") {
        return ReadFaceContourTxtToAnnotatedDatum(labelfile, ori_height, ori_width,
                                       anno_datum);
      } else {
        LOG(FATAL) << "Unknown label file type.";
        return false;
      }
      break;
    default:
      LOG(FATAL) << "Unknown annotation type.";
      return false;
  }
}

bool ReadRichFaceAngleToAnnotatedDatum(const string& filename,
    const string& labelfile, const int height, const int width,
    const int min_dim, const int max_dim, const bool is_color,
    const string& encoding, const AnnoFaceAngleDatum_AnnoType type,
    const string& labeltype, AnnoFaceAngleDatum* anno_datum){
  // Read image to datum.
  bool status = ReadImageToDatum(filename, -1, height, width,
                                 min_dim, max_dim, is_color, encoding,
                                 anno_datum->mutable_datum());
  if (status == false) {
    return status;
  }
  anno_datum->clear_faceangle();
  if (!boost::filesystem::exists(labelfile)) {
    return true;
  }
  // annno type bbox or attributes
  switch (type) {
    case AnnoFaceAngleDatum_AnnoType_FACEANGLE:
      int ori_height, ori_width;
      GetImageSize(filename, &ori_height, &ori_width);
      if (labeltype == "txt") {
        return ReadFaceAngleTxtToAnnotatedDatum(labelfile, ori_height, ori_width,
                                       anno_datum);
      } else {
        LOG(FATAL) << "Unknown label file type.";
        return false;
      }
      break;
    default:
      LOG(FATAL) << "Unknown annotation type.";
      return false;
  }
}

bool ReadRichCcpdToAnnotatedDatum(const string& filename,
    const string& labelfile, const int height, const int width,
    const int min_dim, const int max_dim, const bool is_color,
    const string& encoding, const AnnotatedCCpdDatum_AnnotationType type,
    const string& labeltype,const std::map<string, int>& name_to_label, AnnotatedCCpdDatum* anno_datum){
  // Read image to datum.
  bool status = ReadImageToDatum(filename, -1, height, width,
                                 min_dim, max_dim, is_color, encoding,
                                 anno_datum->mutable_datum());
  if (status == false) {
    return status;
  }
  anno_datum->clear_lpnumber();
  if (!boost::filesystem::exists(labelfile)) {
    return true;
  }
  // annno type bbox or attributes
  switch (type) {
    case AnnotatedCCpdDatum_AnnotationType_CCPD:
      int ori_height, ori_width;
      GetImageSize(filename, &ori_height, &ori_width);
      if (labeltype == "txt") {
        return ReadccpdTxtToAnnotatedDatum(labelfile, ori_height, ori_width,
                                       name_to_label, anno_datum);
      } else {
        LOG(FATAL) << "Unknown label file type.";
        return false;
      }
      break;
    default:
      LOG(FATAL) << "Unknown annotation type.";
      return false;
  }
}

bool ReadRichBlurToAnnotatedDatum(const string& filename,
    const string& labelfile, const int height, const int width,
    const int min_dim, const int max_dim, const bool is_color,
    const string& encoding, const AnnoBlurDatum_AnnoType type,
    const string& labeltype, const std::map<string, int>& name_to_label, 
    AnnoBlurDatum* anno_datum){
  // Read image to datum.
  bool status = ReadImageToDatum(filename, -1, height, width,
                                 min_dim, max_dim, is_color, encoding,
                                 anno_datum->mutable_datum());
  if (status == false) {
    return status;
  }
  anno_datum->clear_faceatti();
  if (!boost::filesystem::exists(labelfile)) {
    return true;
  }
  // annno type bbox or attributes
  switch (type) {
    case AnnoBlurDatum_AnnoType_FACEBLUR:
      int ori_height, ori_width;
      GetImageSize(filename, &ori_height, &ori_width);
      if (labeltype == "txt") {
        return ReadBlurTxtToAnnotatedDatum(labelfile, ori_height, ori_width,
                                       name_to_label, anno_datum);
      } else {
        LOG(FATAL) << "Unknown label file type.";
        return false;
      }
      break;
    default:
      LOG(FATAL) << "Unknown annotation type.";
      return false;
  }
}

#endif  // USE_OPENCV

bool ReadFileToDatum(const string& filename, const int label,
    Datum* datum) {
  std::streampos size;

  fstream file(filename.c_str(), ios::in|ios::binary|ios::ate);
  if (file.is_open()) {
    size = file.tellg();
    std::string buffer(size, ' ');
    file.seekg(0, ios::beg);
    file.read(&buffer[0], size);
    file.close();
    datum->set_data(buffer);
    datum->set_label(label);
    datum->set_encoded(true);
    return true;
  } else {
    return false;
  }
}

// Parse VOC/ILSVRC detection annotation.
bool ReadXMLToAnnotatedDatum(const string& labelfile, const int img_height,
    const int img_width, const std::map<string, int>& name_to_label,
    AnnotatedDatum* anno_datum) {
  ptree pt;
  read_xml(labelfile, pt);

  // Parse annotation.
  int width = 0, height = 0;
  try {
    height = pt.get<int>("annotation.size.height");
    width = pt.get<int>("annotation.size.width");
  } catch (const ptree_error &e) {
    LOG(WARNING) << "When parsing " << labelfile << ": " << e.what();
    height = img_height;
    width = img_width;
  }
  LOG_IF(WARNING, height != img_height) << labelfile <<
      " inconsistent image height.";
  LOG_IF(WARNING, width != img_width) << labelfile <<
      " inconsistent image width.";
  CHECK(width != 0 && height != 0) << labelfile <<
      " no valid image width/height.";
  int instance_id = 0;
  BOOST_FOREACH(ptree::value_type &v1, pt.get_child("annotation")) {
    ptree pt1 = v1.second;
    if (v1.first == "objects") {
      Annotation* anno = NULL;
      bool difficult = false;
      ptree object = v1.second;
      int blured = object.get<int>("blur");
      int occlusioned = object.get<int>("occlusion");
      BOOST_FOREACH(ptree::value_type &v2, object.get_child("")) {
        ptree pt2 = v2.second;
        if (v2.first == "name") {
          string name = pt2.data();
          if (name_to_label.find(name) == name_to_label.end()) {
            LOG(FATAL) << "Unknown name: " << name;
          }
          int label = name_to_label.find(name)->second;
          bool found_group = false;
          for (int g = 0; g < anno_datum->annotation_group_size(); ++g) {
            AnnotationGroup* anno_group =
                anno_datum->mutable_annotation_group(g);
            if (label == anno_group->group_label()) {
              if (anno_group->annotation_size() == 0) {
                instance_id = 0;
              } else {
                instance_id = anno_group->annotation(
                    anno_group->annotation_size() - 1).instance_id() + 1;
              }
              anno = anno_group->add_annotation();
              found_group = true;
            }
          }
          if (!found_group) {
            // If there is no such annotation_group, create a new one.
            AnnotationGroup* anno_group = anno_datum->add_annotation_group();
            anno_group->set_group_label(label);
            anno = anno_group->add_annotation();
            instance_id = 0;
          }
          anno->set_instance_id(instance_id++);
        } else if (v2.first == "difficult") {
          difficult = pt2.data() == "1";
        }else if (v2.first == "boundingbox") {
          int xmin = pt2.get("xmin", 0);
          int ymin = pt2.get("ymin", 0);
          int xmax = pt2.get("xmax", 0);
          int ymax = pt2.get("ymax", 0);
          CHECK_NOTNULL(anno);
          LOG_IF(WARNING, xmin > width) << labelfile <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, ymin > height) << labelfile <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, xmax > width) << labelfile <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, ymax > height) << labelfile <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, xmin < 0) << labelfile <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, ymin < 0) << labelfile <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, xmax < 0) << labelfile <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, ymax < 0) << labelfile <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, xmin > xmax) << labelfile <<
              " bounding box irregular.";
          LOG_IF(WARNING, ymin > ymax) << labelfile <<
              " bounding box irregular.";
          // Store the normalized bounding box.
          NormalizedBBox* bbox = anno->mutable_bbox();
          bbox->set_xmin(static_cast<float>(xmin) / width);
          bbox->set_ymin(static_cast<float>(ymin) / height);
          bbox->set_xmax(static_cast<float>(xmax) / width);
          bbox->set_ymax(static_cast<float>(ymax) / height);
          bbox->mutable_faceattrib()->set_blur(blured);
          bbox->mutable_faceattrib()->set_occlusion(occlusioned);
          bbox->set_difficult(difficult);
        }
      }
    }
  }
#if 0
  int group_size = anno_datum->annotation_group_size();
  LOG(INFO)<<"group_size: "<<group_size;
  for(int nn = 0; nn< group_size; nn++)
  {
    const AnnotationGroup anno_group = anno_datum->annotation_group(nn);
    LOG(INFO) << "=============================";
    LOG(INFO) <<"anno_group label: "<<anno_group.group_label();
    int anno_size = anno_group.annotation_size();
    for(int jj=0; jj<anno_size; jj++)
    {
      const Annotation anno = anno_group.annotation(jj);
      LOG(INFO)<< "anno_instance_id: "<<anno.instance_id();
      NormalizedBBox bbox = anno.bbox();
      LOG(INFO) << "bbox->xmin: "<<bbox.xmin()<<" bbox->ymin: "<<bbox.ymin()
                <<" bbox->xmax: "<<bbox.xmax()<<" bbox->ymax: "<<bbox.ymax()
                <<" bbox->blur: "<<bbox.faceattrib().blur()<<" bbox->occlusion: "<<bbox.faceattrib().occlusion()
                <<" bbox->label: "<<bbox.label();
    }
  }
#endif
  return true;
}

// Parse MSCOCO detection annotation.
bool ReadJSONToAnnotatedDatum(const string& labelfile, const int img_height,
    const int img_width, const std::map<string, int>& name_to_label,
    AnnotatedDatum* anno_datum) {
  ptree pt;
  read_json(labelfile, pt);

  // Get image info.
  int width = 0, height = 0;
  try {
    height = pt.get<int>("image.height");
    width = pt.get<int>("image.width");
  } catch (const ptree_error &e) {
    LOG(WARNING) << "When parsing " << labelfile << ": " << e.what();
    height = img_height;
    width = img_width;
  }
  LOG_IF(WARNING, height != img_height) << labelfile <<
      " inconsistent image height.";
  LOG_IF(WARNING, width != img_width) << labelfile <<
      " inconsistent image width.";
  CHECK(width != 0 && height != 0) << labelfile <<
      " no valid image width/height.";

  // Get annotation info.
  int instance_id = 0;
  BOOST_FOREACH(ptree::value_type& v1, pt.get_child("annotation")) {
    Annotation* anno = NULL;
    bool iscrowd = false;
    ptree object = v1.second;
    // Get category_id.
    string name = object.get<string>("category_id");
    if (name_to_label.find(name) == name_to_label.end()) {
      LOG(FATAL) << "Unknown name: " << name;
    }
    int label = name_to_label.find(name)->second;
    bool found_group = false;
    for (int g = 0; g < anno_datum->annotation_group_size(); ++g) {
      AnnotationGroup* anno_group =
          anno_datum->mutable_annotation_group(g);
      if (label == anno_group->group_label()) {
        if (anno_group->annotation_size() == 0) {
          instance_id = 0;
        } else {
          instance_id = anno_group->annotation(
              anno_group->annotation_size() - 1).instance_id() + 1;
        }
        anno = anno_group->add_annotation();
        found_group = true;
      }
    }
    if (!found_group) {
      // If there is no such annotation_group, create a new one.
      AnnotationGroup* anno_group = anno_datum->add_annotation_group();
      anno_group->set_group_label(label);
      anno = anno_group->add_annotation();
      instance_id = 0;
    }
    anno->set_instance_id(instance_id++);

    // Get iscrowd.
    iscrowd = object.get<int>("iscrowd", 0);

    // Get bbox.
    vector<float> bbox_items;
    BOOST_FOREACH(ptree::value_type& v2, object.get_child("bbox")) {
      bbox_items.push_back(v2.second.get_value<float>());
    }
    CHECK_EQ(bbox_items.size(), 4);
    float xmin = bbox_items[0];
    float ymin = bbox_items[1];
    float xmax = bbox_items[0] + bbox_items[2];
    float ymax = bbox_items[1] + bbox_items[3];
    CHECK_NOTNULL(anno);
    LOG_IF(WARNING, xmin > width) << labelfile <<
        " bounding box exceeds image boundary.";
    LOG_IF(WARNING, ymin > height) << labelfile <<
        " bounding box exceeds image boundary.";
    LOG_IF(WARNING, xmax > width) << labelfile <<
        " bounding box exceeds image boundary.";
    LOG_IF(WARNING, ymax > height) << labelfile <<
        " bounding box exceeds image boundary.";
    LOG_IF(WARNING, xmin < 0) << labelfile <<
        " bounding box exceeds image boundary.";
    LOG_IF(WARNING, ymin < 0) << labelfile <<
        " bounding box exceeds image boundary.";
    LOG_IF(WARNING, xmax < 0) << labelfile <<
        " bounding box exceeds image boundary.";
    LOG_IF(WARNING, ymax < 0) << labelfile <<
        " bounding box exceeds image boundary.";
    LOG_IF(WARNING, xmin > xmax) << labelfile <<
        " bounding box irregular.";
    LOG_IF(WARNING, ymin > ymax) << labelfile <<
        " bounding box irregular.";
    // Store the normalized bounding box.
    NormalizedBBox* bbox = anno->mutable_bbox();
    bbox->set_xmin(xmin / width);
    bbox->set_ymin(ymin / height);
    bbox->set_xmax(xmax / width);
    bbox->set_ymax(ymax / height);
    bbox->set_difficult(iscrowd);
  }
  return true;
}

// Parse plain txt detection annotation: label_id, xmin, ymin, xmax, ymax.
bool ReadTxtToAnnotatedDatum(const string& labelfile, const int height,
    const int width, AnnotatedDatum* anno_datum) {
  std::ifstream infile(labelfile.c_str());
  if (!infile.good()) {
    LOG(INFO) << "Cannot open " << labelfile;
    return false;
  }
  int label = 1;
  float xmin, ymin, xmax, ymax;
  while (infile >> xmin >> ymin >> xmax >> ymax) {
    Annotation* anno = NULL;
    int instance_id = 0;
    bool found_group = false;
    for (int g = 0; g < anno_datum->annotation_group_size(); ++g) {
      AnnotationGroup* anno_group = anno_datum->mutable_annotation_group(g);
      if (label == anno_group->group_label()) {
        if (anno_group->annotation_size() == 0) {
          instance_id = 0;
        } else {
          instance_id = anno_group->annotation(
              anno_group->annotation_size() - 1).instance_id() + 1;
        }
        anno = anno_group->add_annotation();
        found_group = true;
      }
    }
    if (!found_group) {
      // If there is no such annotation_group, create a new one.
      AnnotationGroup* anno_group = anno_datum->add_annotation_group();
      anno_group->set_group_label(label);
      anno = anno_group->add_annotation();
      instance_id = 0;
    }
    anno->set_instance_id(instance_id++);
    LOG_IF(WARNING, xmin > width) << labelfile <<
      " bounding box exceeds image boundary.";
    LOG_IF(WARNING, ymin > height) << labelfile <<
      " bounding box exceeds image boundary.";
    LOG_IF(WARNING, xmax > width) << labelfile <<
      " bounding box exceeds image boundary.";
    LOG_IF(WARNING, ymax > height) << labelfile <<
      " bounding box exceeds image boundary.";
    LOG_IF(WARNING, xmin < 0) << labelfile <<
      " bounding box exceeds image boundary.";
    LOG_IF(WARNING, ymin < 0) << labelfile <<
      " bounding box exceeds image boundary.";
    LOG_IF(WARNING, xmax < 0) << labelfile <<
      " bounding box exceeds image boundary.";
    LOG_IF(WARNING, ymax < 0) << labelfile <<
      " bounding box exceeds image boundary.";
    LOG_IF(WARNING, xmin > xmax) << labelfile <<
      " bounding box irregular.";
    LOG_IF(WARNING, ymin > ymax) << labelfile <<
      " bounding box irregular.";
    // Store the normalized bounding box.
    NormalizedBBox* bbox = anno->mutable_bbox();
    bbox->set_xmin(xmin / width);
    bbox->set_ymin(ymin / height);
    bbox->set_xmax(xmax / width);
    bbox->set_ymax(ymax / height);
    bbox->set_difficult(false);
  }
  return true;
}

bool ReadccpdTxtToAnnotatedDatum(const string& labelfile, const int height,
    const int width, const std::map<string, int>& name_to_label,
    AnnotatedCCpdDatum* anno_datum){
  std::ifstream infile(labelfile.c_str());
  std::string lineStr ;
  std::stringstream sstr ;
  if (!infile.good()) {
    LOG(INFO) << "Cannot open " << labelfile;
    return false;
  }
  LOG(INFO)<<labelfile;
  int lpnum_1, lpnum_2, lpnum_3, lpnum_4, lpnum_5, lpnum_6, lpnum_7;
  while (std::getline(infile, lineStr )) {
    sstr << lineStr;
    sstr >> lpnum_1>>lpnum_2 >>lpnum_3 >>lpnum_4 >>lpnum_5 >>lpnum_6 >>lpnum_7;
    #if 1
    LOG(INFO)<<lpnum_1<<" "<<lpnum_2 <<" "<<lpnum_3 <<" "<<lpnum_4 <<" "<<lpnum_5 <<" "<<lpnum_6 <<" "<<lpnum_7;
    #endif
    LicensePlate* anno = NULL;
    anno = anno_datum->mutable_lpnumber();
    string name = "licenseplate";
    if (name_to_label.find(name) == name_to_label.end()) {
            LOG(FATAL) << "Unknown name: " << name;
    }
    anno->set_chichracter(lpnum_1);
    anno->set_engchracter(lpnum_2);
    anno->set_letternum_1(lpnum_3);
    anno->set_letternum_2(lpnum_4);
    anno->set_letternum_3(lpnum_5);
    anno->set_letternum_4(lpnum_6);
    anno->set_letternum_5(lpnum_7);
    #if 1
    LOG(INFO)<<"chi: "<<anno->chichracter()<< " eng: "<<anno->engchracter()<<" let1: "<<anno->letternum_1()
              << " let2: "<<anno->letternum_2()<<" let3: "<<anno->letternum_3()<<" let4: "<<anno->letternum_4()
              <<" let5: "<<anno->letternum_5();
    #endif
  }
  return true;
}


bool ReadBlurTxtToAnnotatedDatum(const string& labelfile, const int height,
    const int width, const std::map<string, int>& name_to_label,
    AnnoBlurDatum* anno_datum){
  std::ifstream infile(labelfile.c_str());
  std::string lineStr ;
  std::stringstream sstr ;
  if (!infile.good()) {
    LOG(INFO) << "Cannot open " << labelfile;
    return false;
  }
  LOG(INFO)<<labelfile;
  int blur, occlu;
  while (std::getline(infile, lineStr )) {
    sstr << lineStr;
    sstr >> blur>>occlu;
    #if 1
    LOG(INFO)<<blur<<" "<<occlu;
    #endif
    FaceAttributes* anno = NULL;
    anno = anno_datum->mutable_faceatti();
    string name = "face";
    if (name_to_label.find(name) == name_to_label.end()) {
            LOG(FATAL) << "Unknown name: " << name;
    }
    anno->set_blur(blur);
    anno->set_occlusion(occlu);
    #if 1
    LOG(INFO)<<"blur: "<<anno->blur()<< " occlusion: "<<anno->occlusion();
    #endif
  }
  return true;
}


// Parse plain txt detection annotation: label_id, xmin, ymin, xmax, ymax.
bool ReadumdfaceTxtToAnnotatedDatum(const string& labelfile, const int height,
    const int width, AnnoFaceAttributeDatum* anno_datum) {
  std::ifstream infile(labelfile.c_str());
  std::string lineStr ;
  std::stringstream sstr ;
  if (!infile.good()) {
    LOG(INFO) << "Cannot open " << labelfile;
    return false;
  }
  LOG(INFO)<<labelfile;
  float x1, x2, x3, x4, x5, y1, y2, y3, y4, y5, vis_point_1, vis_point_2, vis_point_3, vis_point_4, vis_point_5;
  float yaw, pitch, roll;
  float pr_female, pr_male;
  int booGlass;
  while (std::getline(infile, lineStr )) {
    sstr << lineStr;
    sstr >> x1 >> x2 >> x3 >> x4 >> x5 >> y1 >> y2 >> y3 >> y4 >> y5 >> vis_point_1 >> vis_point_2 >> vis_point_3
        >> vis_point_4 >> vis_point_5 >> yaw >> pitch >> roll >> pr_female >> pr_male >> booGlass;
    float xf1 = float(x1/width), yf1 = float(y1/height);float xf2 = float(x2/width), yf2 = float(y2/height);
    float xf3 = float(x3/width), yf3 = float(y3/height);float xf4 = float(x4/width), yf4 = float(y4/height);
    float xf5 = float(x5/width), yf5 = float(y5/height);
    LOG(INFO)<<"origin height: "<< height << " origin width:  "<<width << " point: " << x1 <<" "<<y1<<" "
             << x2 <<" "<< y2<<" "<< x3 <<" "<< y3 <<" "<< x4 <<" " << y4 <<" "<< x5 <<" "<< y5 <<" " 
             << x6 <<" "<< y6<<" "<< x7 <<" "<< y7 <<" "<< x8 <<" "<< y8 <<" "<< x9 <<" "<< y9 <<" "
             << x10 <<" "<< y10 <<" "<< x11 <<" "<<y11 << " "<< x12 <<" "<< y12 <<" "<< x13 <<" "<< y13 <<" "
             << x14 <<" " << y14 <<" "<< x15 <<" " << y15 <<" "<< x16 <<" "<< y16 <<" "<< x17 <<" "<< y17 <<" "
             << x18 <<" "<< y18 <<" "<< x19 <<" " << y19 <<" "<< x20 <<" " << y20 <<" "<< x21<<" "<<y21 << " "<< yaw <<" "<< pitch <<" "<< roll;
    AnnoFaceAttribute* anno = NULL;
    anno = anno_datum->mutable_facepose();
    // Store the normalized bounding box.
    AnnoFaceLandmarks* landface = anno->mutable_landMark();
    AnnoFaceOritation* faceOri = anno->mutable_faceoritation();
    faceOri->set_yaw(float(yaw));
    faceOri->set_pitch(float(pitch));
    faceOri->set_roll(float(roll));
    landface->mutable_leftEye()->set_x(xf1);
    landface->mutable_leftEye()->set_y(yf1);
    landface->mutable_rightEye()->set_x(xf2);
    landface->mutable_rightEye()->set_y(yf2);
    landface->mutable_nose()->set_x(xf3);
    landface->mutable_nose()->set_y(yf3);
    landface->mutable_leftmouth()->set_x(xf4);
    landface->mutable_leftmouth()->set_y(yf4);
    landface->mutable_point_5()->set_x(xf5);
    landface->mutable_point_5()->set_y(yf5);
    sstr.clear();
  }
  infile.close();
  return true;
}

// Parse plain txt detection annotation: label_id, xmin, ymin, xmax, ymax.
bool ReadFaceContourTxtToAnnotatedDatum(const string& labelfile, const int height,
    const int width, AnnoFaceContourDatum* anno_datum) {
  std::ifstream infile(labelfile.c_str());
  std::string lineStr ;
  std::stringstream sstr ;
  if (!infile.good()) {
    LOG(INFO) << "Cannot open " << labelfile;
    return false;
  }
  float x1, x2, x3, x4, x5; 
  float y1, y2, y3, y4, y5;
  while (std::getline(infile, lineStr )) {
    sstr << lineStr;
    sstr >> x1 >> x2 >> x3 >> x4 >> x5 >> y1 >> y2 >> y3 >> y4 >> y5;
    // Store the normalized bounding box.
    AnnoFaceLandmarks* landface = anno_datum->mutable_facecontour();
    landface->mutable_leftEye()->set_x(float(x1/width));
    landface->mutable_leftEye()->set_y(float(y1/height));
    landface->mutable_rightEye()->set_x(float(x2/width));
    landface->mutable_rightEye()->set_y(float(y2/height));
    landface->mutable_nose()->set_x(float(x3/width));
    landface->mutable_nose()->set_y(float(y3/height));
    landface->mutable_leftmouth()->set_x(float(x4/width));
    landface->mutable_leftmouth()->set_y(float(y4/height));
    landface->mutable_point_5()->set_x(float(x5/width));
    landface->mutable_point_5()->set_y(float(y5/height));
    sstr.clear();
  }
  infile.close();
  return true;
}

// Parse plain txt detection annotation: label_id, xmin, ymin, xmax, ymax.
bool ReadFaceAngleTxtToAnnotatedDatum(const string& labelfile, const int height,
    const int width, AnnoFaceAngleDatum* anno_datum) {
  std::ifstream infile(labelfile.c_str());
  std::string lineStr ;
  std::stringstream sstr ;
  if (!infile.good()) {
    LOG(INFO) << "Cannot open " << labelfile;
    return false;
  }
  float yaw, pitch, roll;
  while (std::getline(infile, lineStr )) {
    sstr << lineStr;
    sstr >> yaw >> pitch >> roll;
    LOG(INFO)<< " "<< yaw <<" "<< pitch <<" "<< roll;
    AnnoFaceOritation* faceOri = anno_datum->mutable_faceangle();
    faceOri->set_yaw(float(yaw));
    faceOri->set_pitch(float(pitch));
    faceOri->set_roll(float(roll));
    sstr.clear();
  }
  infile.close();
  return true;
}


bool ReadFaceAttriTxtToAnnotatedDatum(const string& labelfile, const int height,
    const int width, AnnoFaceDatum* anno_datum) {
  std::ifstream infile(labelfile.c_str());
  std::string lineStr ;
  std::stringstream sstr ;
  if (!infile.good()) {
    LOG(INFO) << "Cannot open " << labelfile;
    return false;
  }
  LOG(INFO)<<labelfile;
  float x1, x2, x3, x4, x5, y1, y2, y3, y4, y5;
  float x11, x22, x33, x44, x55, y11, y22, y33, y44, y55;
  int gender;
  int glass;
  while (std::getline(infile, lineStr )) {
    sstr << lineStr;
    sstr >> x1 >> x2 >> x3 >> x4 >> x5 >> y1 >> y2 >> y3 >> y4 >> y5
          >> gender >> glass;
    #if 0
    LOG(INFO)<< x1 <<" "<< x2 <<" "<< x3 <<" "<< x4 <<" " << x5 <<" "<< y1 <<" "<< y2 <<" "
             << y3 <<" "<< y4 <<" "<< y5 <<" "<< gender <<" "<< glass;
    #endif
    AnnotationFace* anno = NULL;
    anno = anno_datum->mutable_annoface();
    LOG_IF(WARNING, x1 > width) << labelfile <<
      " bounding box exceeds image boundary.";
    LOG_IF(WARNING, y1 > height) << labelfile <<
      " bounding box exceeds image boundary.";
    LOG_IF(WARNING, x2 > width) << labelfile <<
      " bounding box exceeds image boundary.";
    LOG_IF(WARNING, y2 > height) << labelfile <<
      " bounding box exceeds image boundary.";
    LOG_IF(WARNING, x3 > width) << labelfile <<
      " bounding box exceeds image boundary.";
    LOG_IF(WARNING, y3 > height) << labelfile <<
      " bounding box exceeds image boundary.";
    LOG_IF(WARNING, x4 > width) << labelfile <<
      " bounding box exceeds image boundary.";
    LOG_IF(WARNING, y4 > height) << labelfile <<
      " bounding box exceeds image boundary.";
    LOG_IF(WARNING, x5 > width) << labelfile <<
      " bounding box exceeds image boundary.";
    LOG_IF(WARNING, y5 > height) << labelfile <<
      " bounding box exceeds image boundary.";
    LOG_IF(WARNING, x1 < 0) << labelfile <<
      " bounding box exceeds image boundary.";
    LOG_IF(WARNING, y1 < 0) << labelfile <<
      " bounding box exceeds image boundary.";
    LOG_IF(WARNING, x2 < 0) << labelfile <<
      " bounding box exceeds image boundary.";
    LOG_IF(WARNING, x2 < 0) << labelfile <<
      " bounding box exceeds image boundary.";
    // Store the normalized bounding box.
    LandmarkFace* landface = anno->mutable_markface();
    anno->set_gender(gender - 1);
    anno->set_glasses(glass -1);
    x11 = float(x1/width);y11 = float(y1/height);
    x22 = float(x2/width);y22 = float(y2/height);
    x33 = float(x3/width);y33 = float(y3/height);
    x44 = float(x4/width);y44 = float(y4/height);
    x55 = float(x5/width);y55 = float(y5/height);
    #if 1
    LOG(INFO)<< x11 <<" "<< x22 <<" "<< x33 <<" "<< x44 <<" " << x55 <<" "<< y11 <<" "<< y22 <<" "
             << y33 <<" "<< y44 <<" "<< y55 <<" "<< gender <<" "<< glass;
    #endif
    landface->set_x1(x11);
    landface->set_x2(x22);
    landface->set_x3(x33);
    landface->set_x4(x44);
    landface->set_x5(x55);
    landface->set_y1(y11);
    landface->set_y2(y22);
    landface->set_y3(y33);
    landface->set_y4(y44);
    landface->set_y5(y55);
    sstr.clear();
  }
  infile.close();
  return true;
}

bool ReadLabelFileToLabelMap(const string& filename, bool include_background,
    const string& delimiter, LabelMap* map) {
  // cleanup
  map->Clear();

  std::ifstream file(filename.c_str());
  string line;
  // Every line can have [1, 3] number of fields.
  // The delimiter between fields can be one of " :;".
  // The order of the fields are:
  //  name [label] [display_name]
  //  ...
  int field_size = -1;
  int label = 0;
  LabelMapItem* map_item;
  // Add background (none_of_the_above) class.
  if (include_background) {
    map_item = map->add_item();
    map_item->set_name("none_of_the_above");
    map_item->set_label(label++);
    map_item->set_display_name("background");
  }
  while (std::getline(file, line)) {
    vector<string> fields;
    fields.clear();
    boost::split(fields, line, boost::is_any_of(delimiter));
    if (field_size == -1) {
      field_size = fields.size();
    } else {
      CHECK_EQ(field_size, fields.size())
          << "Inconsistent number of fields per line.";
    }
    map_item = map->add_item();
    map_item->set_name(fields[0]);
    switch (field_size) {
      case 1:
        map_item->set_label(label++);
        map_item->set_display_name(fields[0]);
        break;
      case 2:
        label = std::atoi(fields[1].c_str());
        map_item->set_label(label);
        map_item->set_display_name(fields[0]);
        break;
      case 3:
        label = std::atoi(fields[1].c_str());
        map_item->set_label(label);
        map_item->set_display_name(fields[2]);
        break;
      default:
        LOG(FATAL) << "The number of fields should be [1, 3].";
        break;
    }
  }
  return true;
}

bool MapNameToLabel(const LabelMap& map, const bool strict_check,
    std::map<string, int>* name_to_label) {
  // cleanup
  name_to_label->clear();

  for (int i = 0; i < map.item_size(); ++i) {
    const string& name = map.item(i).name();
    const int label = map.item(i).label();
    if (strict_check) {
      if (!name_to_label->insert(std::make_pair(name, label)).second) {
        LOG(FATAL) << "There are many duplicates of name: " << name;
        return false;
      }
    } else {
      (*name_to_label)[name] = label;
    }
  }
  return true;
}

bool MapLabelToName(const LabelMap& map, const bool strict_check,
    std::map<int, string>* label_to_name) {
  // cleanup
  label_to_name->clear();

  for (int i = 0; i < map.item_size(); ++i) {
    const string& name = map.item(i).name();
    const int label = map.item(i).label();
    if (strict_check) {
      if (!label_to_name->insert(std::make_pair(label, name)).second) {
        LOG(FATAL) << "There are many duplicates of label: " << label;
        return false;
      }
    } else {
      (*label_to_name)[label] = name;
    }
  }
  return true;
}

bool MapLabelToDisplayName(const LabelMap& map, const bool strict_check,
    std::map<int, string>* label_to_display_name) {
  // cleanup
  label_to_display_name->clear();

  for (int i = 0; i < map.item_size(); ++i) {
    const string& display_name = map.item(i).display_name();
    const int label = map.item(i).label();
    if (strict_check) {
      if (!label_to_display_name->insert(
              std::make_pair(label, display_name)).second) {
        LOG(FATAL) << "There are many duplicates of label: " << label;
        return false;
      }
    } else {
      (*label_to_display_name)[label] = display_name;
    }
  }
  return true;
}

#ifdef USE_OPENCV
cv::Mat DecodeDatumToCVMatNative(const Datum& datum) {
  cv::Mat cv_img;
  CHECK(datum.encoded()) << "Datum not encoded";
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  cv_img = cv::imdecode(vec_data, -1);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not decode datum ";
  }
  return cv_img;
}
cv::Mat DecodeDatumToCVMat(const Datum& datum, bool is_color) {
  cv::Mat cv_img;
  CHECK(datum.encoded()) << "Datum not encoded";
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv_img = cv::imdecode(vec_data, cv_read_flag);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not decode datum ";
  }
  return cv_img;
}

// If Datum is encoded will decoded using DecodeDatumToCVMat and CVMatToDatum
// If Datum is not encoded will do nothing
bool DecodeDatumNative(Datum* datum) {
  if (datum->encoded()) {
    cv::Mat cv_img = DecodeDatumToCVMatNative((*datum));
    CVMatToDatum(cv_img, datum);
    return true;
  } else {
    return false;
  }
}
bool DecodeDatum(Datum* datum, bool is_color) {
  if (datum->encoded()) {
    cv::Mat cv_img = DecodeDatumToCVMat((*datum), is_color);
    CVMatToDatum(cv_img, datum);
    return true;
  } else {
    return false;
  }
}

void EncodeCVMatToDatum(const cv::Mat& cv_img, const string& encoding,
                        Datum* datum) {
  std::vector<uchar> buf;
  cv::imencode("."+encoding, cv_img, buf);
  datum->set_data(std::string(reinterpret_cast<char*>(&buf[0]),
                              buf.size()));
  datum->set_channels(cv_img.channels());
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->set_encoded(true);
}

void CVMatToDatum(const cv::Mat& cv_img, Datum* datum) {
  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";
  datum->set_channels(cv_img.channels());
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->clear_data();
  datum->clear_float_data();
  datum->set_encoded(false);
  int datum_channels = datum->channels();
  int datum_height = datum->height();
  int datum_width = datum->width();
  int datum_size = datum_channels * datum_height * datum_width;
  std::string buffer(datum_size, ' ');
  for (int h = 0; h < datum_height; ++h) {
    const uchar* ptr = cv_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < datum_width; ++w) {
      for (int c = 0; c < datum_channels; ++c) {
        int datum_index = (c * datum_height + h) * datum_width + w;
        buffer[datum_index] = static_cast<char>(ptr[img_index++]);
      }
    }
  }
  datum->set_data(buffer);
}
#endif  // USE_OPENCV
}  // namespace caffe
