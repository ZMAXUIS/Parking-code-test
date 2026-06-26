import argparse
import logging
# # carla 0.9.11
import sys
sys.path.insert(0,'/-your path-/CARLA_0.9.11/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg')

import carla
import pygame
import time
import os


from data_generation.network_evaluator import NetworkEvaluator
from data_generation.keyboard_control import KeyboardControl
from agent.parking_agent import ParkingAgent, show_control_info
from utils.config import get_train_config_obj


def wait_for_world_ready(world, timeout=30):

    start = time.time()
    while True:
        try:
            world.tick()
            break
        except RuntimeError:
            if time.time() - start > timeout:
                raise RuntimeError("Timeout waiting for CARLA world to be ready")
            time.sleep(0.5)
##
def game_loop(args):
    pygame.init()
    pygame.font.init()
    network_evaluator = None
    rl_process = None
    rl_log = None

    # Online RL toggle (evaluate-time guidance). Put early for visibility.
    use_online_rl = bool(getattr(args, 'use_online_rl', False))

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(60.0)

        # If online RL integration is requested, spawn RL script early
        if use_online_rl:
            import subprocess
            logging.info('Online RL enabled (use_online_rl=%s). Starting RL guide process.', use_online_rl)
            rl_cmd = [sys.executable, os.path.join(os.getcwd(), 'tools', 'rl_finetune_sb3.py')]
            os.makedirs(args.eva_result_path, exist_ok=True)
            rl_log_path = os.path.join(args.eva_result_path, 'rl_online.log')
            rl_log = open(rl_log_path, 'a')
            env = os.environ.copy()
            env['RL_USE_CARLA'] = '1'
            env['RL_CONFIG_PATH'] = args.model_config_path
            env['RL_MODE'] = 'online'
            try:
                rl_process = subprocess.Popen(rl_cmd, stdout=rl_log, stderr=subprocess.STDOUT, env=env)
                logging.info('Started online RL subprocess (pid=%s), logging to %s', rl_process.pid, rl_log.name)
            except Exception as e:
                logging.exception('Failed to start online RL subprocess: %s', e)

        # Robust map selection and loading
        # If user passes '--map train' we treat it as the default evaluation town (change below if needed).
        requested_map = args.map
        if requested_map == 'train':
            # default alias -> change here if you want a different default map
            requested_map = 'Town04_Opt'

        logging.info('Requested map: %s', requested_map)

        # Try to get available maps from the server and fall back if requested map is missing
        chosen_map = requested_map
        try:
            available_maps = client.get_available_maps()
            logging.debug('CARLA server available maps: %s', available_maps)
            if chosen_map not in available_maps:
                logging.warning('Requested map "%s" not available on CARLA server.', chosen_map)
                # Try common fallbacks (prefer Town04 variants).
                # Note: CARLA may return full asset paths like '/Game/Carla/Maps/Town05_Opt', so
                # we match by suffix. We prefer Town04 over Town05 when Town04_Opt is not present.
                candidates = ['Town04_Opt', 'Town04', 'Town04HD', 'Town04_clean']
                found = False
                for candidate in candidates:
                    for am in available_maps:
                        # match either exact name or suffix path
                        if am == candidate or am.endswith('/' + candidate) or am.endswith(candidate):
                            chosen_map = am
                            logging.info('Falling back to available map: %s', chosen_map)
                            found = True
                            break
                    if found:
                        break
                else:
                    # Try explicit 'Town04' before falling back to first available map (user preference)
                    # First try explicit 'Town04' (preferred) — this may succeed if server accepts name
                    logging.info('No Town04 candidates found in server list; trying explicit "Town04" as user-preferred fallback')
                    chosen_map = 'Town04'
                    # Note: if this load fails later, code will surface an error; otherwise it will use Town04.
                    # If you prefer automatic fallback to the server's first available map instead, change here.
                    # as last resort, if Town04 doesn't work later, we will not auto-select Town05_Opt here to respect user preference.
        except Exception as e:
            logging.debug('Could not query available maps from CARLA server (%s). Will try to load requested map directly.', e)

        logging.info('Load Map %s', chosen_map)
        try:
            carla_world = client.load_world(chosen_map)
            # Ensure the world is ready for tick operations (some CARLA versions need a tick after load)
            try:
                wait_for_world_ready(carla_world, timeout=30)
            except Exception:
                # non-fatal: if wait_for_world_ready fails, continue and let subsequent calls raise if necessary
                logging.debug('wait_for_world_ready failed or timed out; continuing and relying on CARLA server state.')
        except RuntimeError as e:
            # Re-raise with more context
            logging.exception('Failed to load map "%s": %s', chosen_map, e)
            raise RuntimeError(f'Map not found or error loading map "{chosen_map}". Server may not have the map installed or CARLA server is not running the expected maps.') from e

        # If RL integration is requested (either via CLI --use_rl or YAML use_rl_finetune), spawn RL script
        # Read training config to check YAML flag
        try:
            train_cfg = get_train_config_obj(args.model_config_path)
        except Exception:
            train_cfg = None

        yaml_use_rl = bool(getattr(train_cfg, 'use_rl_finetune', False)) if train_cfg is not None else False
        cli_use_rl = bool(getattr(args, 'use_rl', False))
        if yaml_use_rl or cli_use_rl:
            import subprocess
            rl_mode = getattr(args, 'rl_mode', 'train')
            logging.info('RL integration requested (cli=%s yaml=%s), mode=%s', cli_use_rl, yaml_use_rl, rl_mode)
            # prepare command: tools/rl_finetune_sb3.py will read config file itself; pass model_config_path via env
            rl_cmd = [sys.executable, os.path.join(os.getcwd(), 'tools', 'rl_finetune_sb3.py')]
            # ensure result dir exists
            os.makedirs(args.eva_result_path, exist_ok=True)
            rl_log_path = os.path.join(args.eva_result_path, f'rl_{rl_mode}.log')
            rl_log = open(rl_log_path, 'a')
            env = os.environ.copy()
            # pass whether to use carla to the RL script via env var
            env['RL_USE_CARLA'] = '1'
            # also pass model config path
            env['RL_CONFIG_PATH'] = args.model_config_path
            try:
                rl_process = subprocess.Popen(rl_cmd, stdout=rl_log, stderr=subprocess.STDOUT, env=env)
                logging.info('Started RL subprocess (pid=%s), logging to %s', rl_process.pid, rl_log.name)
            except Exception as e:
                logging.exception('Failed to start RL subprocess: %s', e)

        carla_world.unload_map_layer(carla.MapLayer.ParkedVehicles)

        network_evaluator = NetworkEvaluator(carla_world, args)
        parking_agent = ParkingAgent(network_evaluator, args)
        controller = KeyboardControl(network_evaluator.world)

        display = pygame.display.set_mode((args.width, args.height),
                                          pygame.HWSURFACE | pygame.DOUBLEBUF)

        steer_wheel_img = pygame.image.load("./resource/steer_wheel.png")
        steer_wheel_img = pygame.transform.scale(steer_wheel_img, (100, 100))
        font = pygame.font.Font(None, 25)

        clock = pygame.time.Clock()
        while True:
            network_evaluator.world_tick()
            # carla 0.9.15
            # network_evaluator.world_tick(clock)

            clock.tick_busy_loop(60)
            if controller.parse_events(client, network_evaluator.world, clock):
                return
            parking_agent.tick()
            network_evaluator.tick(clock)
            network_evaluator.render(display)
            show_control_info(display, parking_agent.get_eva_control(), steer_wheel_img,
                              args.width, args.height, font)
            pygame.display.flip()

    finally:
        # stop recorder if client was created
        if 'client' in locals() and network_evaluator:
            try:
                client.stop_recorder()
            except Exception:
                pass

        # terminate RL subprocess if it was started
        try:
            if rl_process is not None:
                logging.info('Terminating RL subprocess (pid=%s)', rl_process.pid)
                rl_process.terminate()
                rl_process.wait(timeout=5)
        except Exception:
            pass
        # close RL log file if opened
        try:
            if rl_log is not None:
                rl_log.close()
        except Exception:
            pass

        if network_evaluator is not None:
            network_evaluator.destroy()

        pygame.quit()


def str2bool(v):
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Data Generation')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='860x480',
        help='window resolution (default: 860x480)')
    argparser.add_argument(
        '--gamma',
        default=0.0,
        type=float,
        help='Gamma correction of the camera (default: 0.0)')
    argparser.add_argument(
        '--model_path',
        default='./ckpt/last.ckpt',
        help='path to model.ckpt')
    argparser.add_argument(
        '--model_config_path',
        default='./config/training_real.yaml',
        help='path to model training_real.yaml')
    argparser.add_argument(
        '--eva_epochs',
        default=4,
        type=int,
        help='number of eva epochs (default: 4')
    argparser.add_argument(
        '--eva_task_nums',
        default=16,
        type=int,
        help='number of parking slot task (default: 16')
    argparser.add_argument(
        '--eva_parking_nums',
        default=6,
        type=int,
        help='number of parking nums for every slot (default: 6')
    argparser.add_argument(
        '--map',
        default='train',
        help='map of carla (default: train). Use map name like "Town04" or "Town04_Opt". If set to "train" it maps to Town04 by default. You can change the default mapping in the game_loop code.')
    argparser.add_argument(
        '--shuffle_veh',
        default=True,
        type=str2bool,
        help='shuffle static vehicles between tasks (default: True)')
    argparser.add_argument(
        '--shuffle_weather',
        default=False,
        type=str2bool,
        help='shuffle weather between tasks (default: False)')
    argparser.add_argument(
        '--random_seed',
        default=0,
        help='random seed to initialize env; if sets to 0, use current timestamp as seed (default: 0)')
    argparser.add_argument(
        '--bev_render_device',
        default='cpu',
        help='device used for BEV Rendering (default: cpu)',
        choices=['cpu', 'cuda'])
    argparser.add_argument(
        '--show_eva_imgs',
        default=False,
        type=str2bool,
        help='show eva figure in eva model (default: False)')
    argparser.add_argument(
        '--eva_result_path',
        default='./eva_result',
        help='path to save eva csv file')
    argparser.add_argument(
        '--use_rl',
        default=False,
        type=str2bool,
        help='whether to use RL integration (default: False)')
    argparser.add_argument(
        '--use_online_rl',
        default=False,
        type=str2bool,
        help='whether to use online RL guidance during evaluation (default: False)')
    argparser.add_argument(
        '--rl_mode',
        default='train',
        help='RL mode, either "train" or "eval" (default: "train")',
        choices=['train', 'eval'])
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    try:
        game_loop(args)

    except KeyboardInterrupt:
        logging.info('Cancelled by user. Bye!')


if __name__ == '__main__':
    main()

