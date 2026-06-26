import argparse
import rospy
import os
import subprocess
import yaml

import utils.fix_libtiff
from model_interface.model_interface import get_parking_model
from utils.config import get_inference_config_obj
from utils.ros_interface import RosInterface
from utils.evaluation import EvaluationManager


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--inference_config_path', default="./config/inference_real.yaml", type=str)
    arg_parser.add_argument('--eval_bag_path', default=None, type=str, help='rosbag path for evaluation (optional)')
    arg_parser.add_argument('--eval_out_dir', default=None, type=str, help='output directory for evaluation results (optional)')
    arg_parser.add_argument('--run_config', default=None, type=str, help='optional YAML file to specify demo and eval params')
    args = arg_parser.parse_args()

    # If a run_config YAML is provided, load it and override parameters
    if args.run_config is not None and os.path.exists(args.run_config):
        with open(args.run_config, 'r') as f:
            cfg = yaml.safe_load(f)
        # expected keys: run_demo (bool), demo_scenario (str/int), demo_script (path), inference_config_path, eval_bag_path, eval_out_dir
        run_demo = cfg.get('run_demo', False)
        demo_scenario = cfg.get('demo_scenario', '1')
        demo_script = cfg.get('demo_script', './demo_scene/demo.sh')
        # override args
        args.inference_config_path = cfg.get('inference_config_path', args.inference_config_path)
        args.eval_bag_path = cfg.get('eval_bag_path', args.eval_bag_path)
        args.eval_out_dir = cfg.get('eval_out_dir', args.eval_out_dir)

        if run_demo:
            # launch demo script in background
            demo_cmd = ["sh", os.path.join(os.getcwd(), demo_script), str(demo_scenario)]
            try:
                subprocess.Popen(demo_cmd)
                print(f"Started demo script: {' '.join(demo_cmd)}")
            except Exception as e:
                print(f"Failed to start demo script {demo_cmd}: {e}")

    rospy.init_node("e2e_traj_pred")

    ros_interface_obj = RosInterface()
    threads = ros_interface_obj.receive_info()

    inference_cfg = get_inference_config_obj(args.inference_config_path)

    ParkingInferenceModelModule = get_parking_model(data_mode=inference_cfg.train_meta_config.data_mode, run_mode="inference")
    parking_inference_obj = ParkingInferenceModelModule(inference_cfg, ros_interface_obj=ros_interface_obj, eval_bag_path=args.eval_bag_path, eval_out_dir=args.eval_out_dir)
    parking_inference_obj.predict(mode=inference_cfg.predict_mode)

    for thread in threads:
        thread.join()