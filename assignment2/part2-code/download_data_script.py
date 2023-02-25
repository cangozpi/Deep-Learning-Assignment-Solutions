import requests
from PIL import Image
import os
import threading
from threading import Thread, Lock

# Parameters ====================
timeout = 5
saved_img_folder_path = './part2-faceDataset'
max_thread_workers = 50
# ===============================


# Helper functions
def extract_info(line):
    parsed_line = line.split('\t')
    full_name = parsed_line[0]

    img_url = parsed_line[3]

    face_bounding_box_coords = parsed_line[4]
    face_bounding_box_coords = face_bounding_box_coords.split(',')
    face_bounding_box_coords = list(map((lambda x:  int(x)), iter(face_bounding_box_coords)))

    img_hash = parsed_line[5]

    return full_name, img_url, face_bounding_box_coords, img_hash


def download_img(img_url, face_bounding_box_coords, img_count):
    global successfull_count
    global failure_count
    global successfull_count_lock
    global failure_count_lock
    try:
        img_req = requests.get(img_url, timeout=timeout, stream = True)
        if img_req.status_code == 200:
            img = Image.open(img_req.raw)
            # Crop the face from the image
            img = img.crop(face_bounding_box_coords)
            # unique img name
            img_file_name = f'{full_name}-{img_count}.jpg'

            #img.show()
            img.save(os.path.join(saved_img_folder_path, img_file_name))

            # check SHA-256 hashes TODO:
            #readable_hash = hashlib.sha256(img.tobytes()).hexdigest();
            #print(readable_hash)
            successfull_count_lock.acquire()
            successfull_count += 1
            successfull_count_lock.release()
        
    except:
        failure_count_lock.acquire()
        failure_count += 1
        failure_count_lock.release()
        #print(f'Problem while downloading from url: {img_url}')


def asyncDownload(download_img, img_url, face_bounding_box_coords, img_count):
    cur_thread = Thread(target=download_img, args=(img_url, face_bounding_box_coords, img_count))
    cur_thread.start()
    return cur_thread

def checkThreadNums(thread_list):
    global max_thread_workers
    if len(thread_list) == max_thread_workers:
        # Check for all threads to terminate
        for thread in thread_list:
            if thread != None:
                thread.join()
        thread_list = []
    return thread_list


def printInfo():
    global successfull_count
    global failure_count
    global successfull_count_lock
    global failure_count_lock
    
    successfull_count_lock.acquire()
    failure_count_lock.acquire()
    print(f"Successfully saved image number: {successfull_count}, Number of failed attempts: {failure_count}, Iteration: {img_count}/{num_total_imgs}, Number of Threads being used: {threading.active_count()}", end="\r")
    successfull_count_lock.release()
    failure_count_lock.release()



name_set = set()
img_count = 0
successfull_count = 0
failure_count = 0
num_total_imgs = 0
successfull_count_lock = Lock()
failure_count_lock = Lock()
thread_list = []
for subset_file in ['./part2-data/subset_actors.txt', './part2-data/subset_actresses.txt']:
    print(f'Downloading from \"{subset_file}\" into \"{saved_img_folder_path}\"', '-'*20)
    num_total_imgs += len(list(open(subset_file).readlines()))
    for line in open(subset_file).readlines():
        # Extract information
        full_name, img_url, face_bounding_box_coords, img_hash = extract_info(line)
        #print(full_name)
        #print(img_url)
        #print(face_bounding_box_coords)
        #print(img_hash)
    
        # Create image download directory if does not exist
        if not os.path.exists(saved_img_folder_path):
            os.makedirs(saved_img_folder_path)
    
        # Use Multi-threading to speed up image downloads
        thread_list = checkThreadNums(thread_list) # Check/Clean threads
        printInfo()# print info
        cur_thread = asyncDownload(download_img, img_url, face_bounding_box_coords, img_count) # create new thread for current download
        thread_list.append(cur_thread)
        
        img_count += 1

    
