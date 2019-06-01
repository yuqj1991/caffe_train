// This program converts a set of images and annotations to a lmdb/leveldb by
// storing them as AnnotatedDatum proto buffers.
// Usage:
//   convert_annoset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images and
// annotations, and LISTFILE should be a list of files as well as their labels
// or label files.
// For classification task, the file should be in the format as
//   imgfolder1/img1.JPEG 7
//   ....
// For detection task, the file should be in the format as
//   imgfolder1/img1.JPEG annofolder1/anno1.xml
//   ....

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "boost/variant.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

DEFINE_bool(gray, false,
    "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "lmdb",
    "The backend {lmdb, leveldb} for storing the result");
DEFINE_string(anno_type, "classification",
    "The type of annotation {classification, detection, faceattributes, facepose}.");
DEFINE_string(label_type, "xml",
    "The type of annotation file format.{xml , txt}");
DEFINE_string(label_map_file, "",
    "A file with LabelMap protobuf message.");
DEFINE_bool(check_label, false,
    "When this option is on, check that there is no duplicated name/label.");
DEFINE_int32(min_dim, 0,
    "Minimum dimension images are resized to (keep same aspect ratio)");
DEFINE_int32(max_dim, 0,
    "Maximum dimension images are resized to (keep same aspect ratio)");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_bool(check_size, false,
    "When this option is on, check that all the datum have the same size");
DEFINE_bool(encoded, false,
    "When this option is on, the encoded image will be save in datum");
DEFINE_string(encode_type, "",
    "Optional: What type should we encode the image as ('png','jpg',...).");

int main(int argc, char** argv) {
#ifdef USE_OPENCV
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of images and annotations to the "
        "leveldb/lmdb format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_annoset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_annoset");
    return 1;
  }

  const bool is_color = !FLAGS_gray;
  const bool check_size = FLAGS_check_size;
  const bool encoded = FLAGS_encoded;
  const string encode_type = FLAGS_encode_type;
  const string anno_type = FLAGS_anno_type;
  AnnotatedDatum_AnnotationType type;
  AnnotatedDatum_AnnoataionAttriType face_det_attri_type;
  AnnotatedDatum_AnnoataionAttriType ccpd_det_attri_type;
  AnnoFaceDatum_AnnotationType anno_face_attri_type;
  AnnoFacePoseDatum_AnnoType anno_facepose_type;
  AnnoFaceContourDatum_AnnoType anno_facecontour_type;
  AnnoFaceAngleDatum_AnnoType anno_faceangle_type;
  AnnotatedCCpdDatum_AnnotationType anno_ccpd_type;
  const string label_type = FLAGS_label_type;
  const string label_map_file = FLAGS_label_map_file;
  const bool check_label = FLAGS_check_label;
  std::map<std::string, int> name_to_label;

  std::ifstream infile(argv[2]);
  std::vector<std::pair<std::string, boost::variant<int, std::string> > > lines;
  std::string filename;
  int label;
  std::string labelname;
  if (anno_type == "classification") {
    while (infile >> filename >> label) {
      lines.push_back(std::make_pair(filename, label));
    }
  } else if (anno_type == "detection_face") {
    type = AnnotatedDatum_AnnotationType_BBOX;
    face_det_attri_type = AnnotatedDatum_AnnoataionAttriType_FACE;
    LabelMap label_map;
    CHECK(ReadProtoFromTextFile(label_map_file, &label_map))
        << "Failed to read label map file.";
    CHECK(MapNameToLabel(label_map, check_label, &name_to_label))
        << "Failed to convert name to label.";
    while (infile >> filename >> labelname) {
      lines.push_back(std::make_pair(filename, labelname));
    }
  }else if (anno_type == "detection_ccpd") {
    type = AnnotatedDatum_AnnotationType_BBOX;
    ccpd_det_attri_type = AnnotatedDatum_AnnoataionAttriType_NORMALL;
    LabelMap label_map;
    CHECK(ReadProtoFromTextFile(label_map_file, &label_map))
        << "Failed to read label map file.";
    CHECK(MapNameToLabel(label_map, check_label, &name_to_label))
        << "Failed to convert name to label.";
    while (infile >> filename >> labelname) {
      lines.push_back(std::make_pair(filename, labelname));
    }
  }else if(anno_type == "faceattributes") {
		anno_face_attri_type = AnnoFaceDatum_AnnotationType_FACEMARK;
    while (infile >> filename >> labelname) {
      lines.push_back(std::make_pair(filename, labelname));
    }
	} else if(anno_type == "facepose") {
		anno_facepose_type = AnnoFacePoseDatum_AnnoType_FACEPOSE;
    while (infile >> filename >> labelname) {
      lines.push_back(std::make_pair(filename, labelname));
    }
	}else if(anno_type == "facecontour") {
		anno_facecontour_type = AnnoFaceContourDatum_AnnoType_FACECONTOUR;
    while (infile >> filename >> labelname) {
      lines.push_back(std::make_pair(filename, labelname));
    }
	} else if(anno_type == "faceangle") {
		anno_faceangle_type = AnnoFaceAngleDatum_AnnoType_FACEANGLE;
    while (infile >> filename >> labelname) {
      lines.push_back(std::make_pair(filename, labelname));
    }
	} else if (anno_type == "Rec_ccpd") {
    anno_ccpd_type = AnnotatedCCpdDatum_AnnotationType_CCPD;
    LabelMap label_map;
    CHECK(ReadProtoFromTextFile(label_map_file, &label_map))
        << "Failed to read label map file.";
    CHECK(MapNameToLabel(label_map, check_label, &name_to_label))
        << "Failed to convert name to label.";
    while (infile >> filename >> labelname) {
      lines.push_back(std::make_pair(filename, labelname));
    }
  }
  if (FLAGS_shuffle) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    shuffle(lines.begin(), lines.end());
  }
  LOG(INFO) << "A total of " << lines.size() << " images.";

  if (encode_type.size() && !encoded)
    LOG(INFO) << "encode_type specified, assuming encoded=true.";

  int min_dim = std::max<int>(0, FLAGS_min_dim);
  int max_dim = std::max<int>(0, FLAGS_max_dim);
  int resize_height = std::max<int>(0, FLAGS_resize_height);
  int resize_width = std::max<int>(0, FLAGS_resize_width);

  // Create new DB
  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  db->Open(argv[3], db::NEW);
  scoped_ptr<db::Transaction> txn(db->NewTransaction());

  // Storing to db
  std::string root_folder(argv[1]);
  AnnotatedDatum anno_datum;
  Datum* datum = anno_datum.mutable_datum();
  AnnoFaceDatum anno_faceDatum;
  AnnoFacePoseDatum anno_faceposeDatum;
  AnnoFaceContourDatum anno_faceContourDatum;
  AnnoFaceAngleDatum anno_faceAngleDatum;
  AnnotatedCCpdDatum anno_ccpdDatum;
  int count = 0;
  int data_size = 0;
  bool data_size_initialized = false;

  for (int line_id = 0; line_id < lines.size(); ++line_id) {
    bool status = true;
    std::string enc = encode_type;
    if (encoded && !enc.size()) {
      // Guess the encoding type from the file name
      string fn = lines[line_id].first;
      size_t p = fn.rfind('.');
      if ( p == fn.npos )
        LOG(WARNING) << "Failed to guess the encoding of '" << fn << "'";
      enc = fn.substr(p);
      std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
    }
    filename = lines[line_id].first;
    if (anno_type == "classification") {
      label = boost::get<int>(lines[line_id].second);
      status = ReadImageToDatum(filename, label, resize_height, resize_width,
          min_dim, max_dim, is_color, enc, datum);
    } else if (anno_type == "detection_face") {
      labelname = boost::get<std::string>(lines[line_id].second);
      status = ReadRichImageToAnnotatedDatum(filename, labelname, resize_height,
          resize_width, min_dim, max_dim, is_color, enc, type, face_det_attri_type, label_type,
          name_to_label, &anno_datum);
      anno_datum.set_type(AnnotatedDatum_AnnotationType_BBOX);
      anno_datum.set_attri_type(AnnotatedDatum_AnnoataionAttriType_FACE);
    }else if (anno_type == "detection_ccpd") {
      labelname = boost::get<std::string>(lines[line_id].second);
      status = ReadRichImageToAnnotatedDatum(filename, labelname, resize_height,
          resize_width, min_dim, max_dim, is_color, enc, type, ccpd_det_attri_type, label_type,
          name_to_label, &anno_datum);
      anno_datum.set_type(AnnotatedDatum_AnnotationType_BBOX);
      anno_datum.set_attri_type(AnnotatedDatum_AnnoataionAttriType_NORMALL);
    } else if(anno_type == "faceattributes") {
      labelname = boost::get<std::string>(lines[line_id].second); // lines  contain imagename & label.txt
		  status = ReadRichFaceToAnnotatedDatum(filename,
          labelname, resize_height, resize_width, min_dim, max_dim, is_color,
          enc, anno_face_attri_type, label_type, &anno_faceDatum);
      anno_faceDatum.set_type(AnnoFaceDatum_AnnotationType_FACEMARK);
	  } else if(anno_type == "facepose") {
      labelname = boost::get<std::string>(lines[line_id].second); // lines contain imagename & label.txt
		  status = ReadRichFacePoseToAnnotatedDatum(filename,
          labelname, resize_height, resize_width, min_dim, max_dim, is_color,
          enc, anno_facepose_type, label_type, &anno_faceposeDatum);
      anno_faceposeDatum.set_type(AnnoFacePoseDatum_AnnoType_FACEPOSE);
	  }else if(anno_type == "facecontour") {
      labelname = boost::get<std::string>(lines[line_id].second); // lines contain imagename & label.txt
		  status = ReadRichFaceContourToAnnotatedDatum(filename,
          labelname, resize_height, resize_width, min_dim, max_dim, is_color,
          enc, anno_facecontour_type, label_type, &anno_faceContourDatum);
      anno_faceContourDatum.set_type(AnnoFaceContourDatum_AnnoType_FACECONTOUR);
	  }else if(anno_type == "faceangle") {
      labelname = boost::get<std::string>(lines[line_id].second); // lines contain imagename & label.txt
		  status = ReadRichFaceAngleToAnnotatedDatum(filename,
          labelname, resize_height, resize_width, min_dim, max_dim, is_color,
          enc, anno_faceangle_type, label_type, &anno_faceAngleDatum);
      anno_faceAngleDatum.set_type(AnnoFaceAngleDatum_AnnoType_FACEANGLE);
	  }else if (anno_type == "Rec_ccpd") {
      labelname = boost::get<std::string>(lines[line_id].second);
      status = ReadRichCcpdToAnnotatedDatum(filename, labelname, resize_height,
          resize_width, min_dim, max_dim, is_color, enc, anno_ccpd_type, label_type,
          name_to_label, &anno_ccpdDatum);
      anno_ccpdDatum.set_type(AnnotatedCCpdDatum_AnnotationType_CCPD);
    }
    if (status == false) {
      LOG(WARNING) << "Failed to read " << lines[line_id].first;
      continue;
    }
    if (check_size) {
      if (!data_size_initialized) {
        data_size = datum->channels() * datum->height() * datum->width();
        data_size_initialized = true;
      } else {
        const std::string& data = datum->data();
        CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
            << data.size();
      }
    }
    // sequential
    string key_str = caffe::format_int(line_id, 8) + "_" + lines[line_id].first;

    // Put in db
    string out;
    if(anno_type == "classification" || anno_type == "detection_face"|| anno_type == "detection_ccpd")
    {
      CHECK(anno_datum.SerializeToString(&out));
      txn->Put(key_str, out);
    }else if(anno_type == "faceattributes") {
      CHECK(anno_faceDatum.SerializeToString(&out));
      txn->Put(key_str, out);
    }else if(anno_type == "facepose") {
      CHECK(anno_faceposeDatum.SerializeToString(&out));
      txn->Put(key_str, out);
    }else if(anno_type == "facecontour") {
      CHECK(anno_faceContourDatum.SerializeToString(&out));
      txn->Put(key_str, out);
    }else if(anno_type == "faceangle") {
      CHECK(anno_faceAngleDatum.SerializeToString(&out));
      txn->Put(key_str, out);
    }else if(anno_type == "Rec_ccpd") {
      CHECK(anno_ccpdDatum.SerializeToString(&out));
      txn->Put(key_str, out);
    }
    if (++count % 1000 == 0) {
      // Commit db
      txn->Commit();
      txn.reset(db->NewTransaction());
      LOG(INFO) << "Processed " << count << " files.";
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    txn->Commit();
    LOG(INFO) << "Processed " << count << " files.";
  }
#else
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  return 0;
}
