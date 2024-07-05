import json
import numpy as np
import csv

# Load the ground truth data from the JSON file
with open('/home/stefan/PycharmProjects/ZS6D/gts/test_gts/ycbv_bop_test_gt_sam_myset.json', 'r') as file:
    ground_truth_data = json.load(file)

def parse_calculated_data(filename):
    calculated = {}
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header row
        for row in csv_reader:
            scene_id, im_id, obj_id, score, R_str, t_str, time = row
            R = np.array([float(x) for x in R_str.split()]).reshape(3, 3)
            t = np.array([float(x) for x in t_str.split()])

            # Check if object was not found
            not_found = np.allclose(R, np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])) or np.allclose(t, np.zeros(3))

            calculated[(scene_id, obj_id)] = {
                'R': R,
                't': t,
                'score': float(score),
                'time': float(time),
                'found': not not_found
            }
    return calculated

# Calculated values
calculated_data = parse_calculated_data('/home/stefan/PycharmProjects/ZS6D/results/results_ycbv_bop_myset_crocomv2.csv')

'''
Dino
Total Objects: 55
Detected Objects: 50
Detection Rate: 0.91
Average Rotation Error: 1.63 radians
Average Translation Error: 161.69 mm
AR Score: 0.1939
Average Processing Time: 15.1437 seconds

CroCo
Total Objects: 55
Detected Objects: 4
Detection Rate: 0.07
Average Rotation Error: 3.07 radians
Average Translation Error: 344.32 mm
AR Score: 0.0000
Average Processing Time: 0.1558 seconds

CroCoM
Total Objects: 55
Detected Objects: 51
Detection Rate: 0.93
Average Rotation Error: 1.27 radians
Average Translation Error: 187.43 mm
AR Score: 0.2485
Average Processing Time: 9.4332 seconds

CroCoMv2
Total Objects: 55
Detected Objects: 50
Detection Rate: 0.91
Average Rotation Error: 1.20 radians
Average Translation Error: 148.66 mm
AR Score: 0.4030
Average Processing Time: 9.0493 seconds
'''


def rotation_error(R1, R2):
    return np.arccos((np.trace(R1.T @ R2) - 1) / 2)


def translation_error(t1, t2):
    return np.linalg.norm(t1 - t2)


def add_score(error, thresholds):
    return sum([error <= t for t in thresholds]) / len(thresholds)


# AR score thresholds
visib_gt_min = 0.1
error_thresh = {
    'n_top': -1,
    'vis_fract_th': 0.0,
    'err_thresh': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
}

total_objects = 0
detected_objects = 0
rotation_errors = []
translation_errors = []
ar_scores = []
processing_times = []

for scene_key, objects in ground_truth_data.items():
    for obj in objects:
        total_objects += 1
        scene_id = obj['scene_id']
        obj_id = str(obj['obj_id'])
        key = (scene_id, obj_id)

        if key in calculated_data:
            calc_data = calculated_data[key]
            processing_times.append(calc_data['time'])

            if calc_data['found']:
                detected_objects += 1
                gt_R = np.array(obj['cam_R_m2c']).reshape(3, 3)
                gt_t = np.array(obj['cam_t_m2c'])
                calc_R = calc_data['R']
                calc_t = calc_data['t']

                r_error = rotation_error(gt_R, calc_R)
                t_error = translation_error(gt_t, calc_t)
                rotation_errors.append(r_error)
                translation_errors.append(t_error)

                # Calculate AR score
                if obj['visib_fract'] >= visib_gt_min:
                    # Use object diameter for normalization
                    diameter = obj['model_info']['diameter']
                    normalized_error = t_error / diameter
                    ar_score = add_score(normalized_error, error_thresh['err_thresh'])
                    ar_scores.append(ar_score)
            else:
                # Object not found, consider it as maximum error
                rotation_errors.append(np.pi)  # Maximum rotation error
                translation_errors.append(np.inf)  # Maximum translation error
                if obj['visib_fract'] >= visib_gt_min:
                    ar_scores.append(0)  # Minimum AR score

detection_rate = detected_objects / total_objects
avg_rotation_error = np.mean(rotation_errors)
avg_translation_error = np.mean([e for e in translation_errors if e != np.inf])
ar_score = np.mean(ar_scores) if ar_scores else 0
avg_processing_time = np.mean(processing_times)

print(f"Total Objects: {total_objects}")
print(f"Detected Objects: {detected_objects}")
print(f"Detection Rate: {detection_rate:.2f}")
print(f"Average Rotation Error: {avg_rotation_error:.2f} radians")
print(f"Average Translation Error: {avg_translation_error:.2f} mm")
print(f"AR Score: {ar_score:.4f}")
print(f"Average Processing Time: {avg_processing_time:.4f} seconds")