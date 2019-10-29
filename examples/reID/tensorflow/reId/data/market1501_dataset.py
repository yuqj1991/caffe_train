import numpy as np
import h5py
import os
import cv2
import random
import shutil
import tensorflow as tf


def get_pair(path, set, ids, positive):
    pair = []
    pic_name = []
    files = os.listdir('%s/%s' % (path, set))
    if positive:
        value = random.sample(ids, 1)
        id = [str(value[0]), str(value[0])]
    else:
        id = random.sample(ids, 2)
    id = [str(id[0]), str(id[1])]
    for i in range(2):
        # id_files = [f for f in files if (f[0:4] == ('%04d' % id[i]) or (f[0:2] == '-1' and id[i] == -1))]
        id_files = [f for f in files if f.split('_')[0] == id[i]]
        pic_name.append(random.sample(id_files, 1))
    for pic in pic_name:
        pair.append('%s/%s/' % (path, set) + pic[0])

    return pair


def get_id(path, set):
    files = os.listdir('%s/%s' % (path, set))
    IDs = []
    for f in files:
        IDs.append(f.split('_')[0])
    IDs = list(set(IDs))
    return IDs


def read_data(path, set, ids, image_width, image_height, batch_size):
    batch_images = []
    labels = []
    for i in range(batch_size // 2):
        pairs = [get_pair(path, set, ids, True), get_pair(path, set, ids, False)]
        for pair in pairs:
            images = []
            for p in pair:
                image = cv2.imread(p)
                image = cv2.resize(image, (image_width, image_height))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image)
            batch_images.append(images)
        labels.append([1., 0.])
        labels.append([0., 1.])
    return np.transpose(batch_images, (1, 0, 2, 3, 4)), np.array(labels)


def get_list_from_label_file(image_label_file_):
    image_list = []
    label_list = []
    i = 0
    with open(image_label_file_, 'r') as anno_file_:
        for contentline in anno_file_.readlines():
            label = []
            curLine = contentline.strip().split(' ')
            image_list.append(curLine[0] + '.jpg')
            label_list.append(int(curLine[1]))
            anno_file_.close()
    return image_list, label_list


def _parse_image(filename, label):
    """
    ead and pre-process image
    """
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_converted = tf.cast(image_decoded, tf.float32)
    resized_image = tf.image.resize_images(image_converted, [160, 80], method=0)
    return resized_image, label


def train_preprocess(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=32)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    # Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def generate_dataset_softmax(imgpathplace_holder, label_placeholder, batchsize_placeholder):
    dataset = tf.data.Dataset.from_tensor_slices((imgpathplace_holder, label_placeholder))
    dataset = dataset.shuffle(buffer_size=25259)
    dataset = dataset.map(_parse_image)
    dataset = dataset.map(train_preprocess)
    dataset = dataset.batch(batch_size=batchsize_placeholder).repeat()
    iterator = dataset.make_initializable_iterator()
    return iterator


class ImageClass():
    "Stores the paths to images for a given class"

    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)


def generate_name_dir(srcDir, disDir):
    for imgfile in os.listdir(srcDir):
        srcimgfile = srcDir + '/' + imgfile
        class_name = 'n' + imgfile.split('_')[0]
        classDir = disDir + "/" + class_name
        distimgfile = classDir + '/' + imgfile
        if not os.path.isdir(classDir):
            os.makedirs(classDir)
            shutil.copyfile(srcimgfile, distimgfile)
        else:
            for class_id in os.listdir(disDir):
                if class_name == class_id:
                    shutil.copyfile(srcimgfile, distimgfile)


def get_image_paths(Dir):
    image_paths = []
    if os.path.isdir(Dir):
        images = os.listdir(Dir)
        image_paths = [os.path.join(Dir, img) for img in images]
    return image_paths


def get_dataset(path, has_class_directories=True):
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp) \
               if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(ImageClass(class_name, image_paths))

    return dataset


def sample_people(dataset, people_per_batch, images_per_person):
    nrof_images = people_per_batch * images_per_person
    # dataset is a list
    # Sample classes from the dataset
    nrof_classes = len(dataset)
    class_indices = np.arange(nrof_classes)
    np.random.shuffle(class_indices)
    i = 0
    image_paths = []
    num_per_class = []
    sampled_class_indices = []
    # Sample images from these classes until we have enough
    while len(image_paths) < nrof_images:
        class_index = class_indices[i]
        nrof_images_in_class = len(dataset[class_index])
        image_indices = np.arange(nrof_images_in_class)
        np.random.shuffle(image_indices)
        nrof_images_from_class = min(nrof_images_in_class, images_per_person, nrof_images - len(image_paths))
        idx = image_indices[0:nrof_images_from_class]
        # print("nrof_images_in_class: %d, nrof_images_from_class: %d, idx: %d"%(nrof_images_in_class, nrof_images_from_class, len(idx)))
        image_paths_for_class = [dataset[class_index].image_paths[j] for j in idx]
        sampled_class_indices += [class_index] * nrof_images_from_class
        image_paths += image_paths_for_class
        num_per_class.append(nrof_images_from_class)
        i += 1
    return image_paths, num_per_class


def selct_triplet_sample(embeddings, nrof_images_per_class, image_paths, people_per_batch, alpha):
    trip_idx = 0
    emb_start_idx = 0
    triplets = []
    trip_idx = 0
    num_trips = 0
    for i in range(people_per_batch):
        nrof_images = nrof_images_per_class[i]
        for j in range(1, nrof_images):
            a_idx = emb_start_idx + j - 1
            neg_distance_sqr = np.sum(np.square(embeddings[a_idx] - embeddings), axis=1)
            for pair in range(j, nrof_images):
                p_idx = emb_start_idx + pair
                pos_distance_sqr = np.sum(np.square(embeddings[a_idx] - embeddings[p_idx]), axis=1)
                neg_distance_sqr[emb_start_idx: emb_start_idx + nrof_images] = np.NaN
                all_neg = np.where(neg_distance_sqr - pos_distance_sqr < alpha)[0]
                nrof_random_negs = all_neg.shape[0]
                if nrof_random_negs > 0:
                    rnd_idx = np.random.randint(nrof_random_negs)
                    n_idx = all_neg[rnd_idx]
                    triplets.append((image_paths[a_idx], image_paths[p_idx], image_paths[n_idx]))
                    # print('Triplet %d: (%d, %d, %d), pos_dist=%2.6f, neg_dist=%2.6f (%d, %d, %d, %d, %d)' %
                    # (trip_idx, a_idx, p_idx, n_idx, pos_dist_sqr, neg_dists_sqr[n_idx], nrof_random_negs, rnd_idx, i, j, emb_start_idx))
                    trip_idx += 1
                num_trips += 1
        emb_start_idx += nrof_images
    np.random.shuffle(triplets)
    return triplets, num_trips, len(triplets)

# if __name__ == '__main__':
# prepare_data(sys.argv[1])
