import argparse

parser = argparse.ArgumentParser(description='Specify video File by number.')
parser.add_argument('-v', action="store", dest="video_num", help="Video Number", type=int, default=1)
parser.add_argument('-d', action="store", dest="video_delay", help="Video frame delay (ms)", type=int, default=10)
parser.add_argument('-inputs', action="store", dest="video_in_dir", help="Video directory (e.g.: 'data/')", default='data/')
parser.add_argument('-outputs', action="store", dest="video_out_dir", help="Video directory (e.g.: 'out/')", default='out/')
parser.add_argument('-r', action="store", dest="video_rot", help="Video Rotation. (Multiple of 90 degrees)", type=int, default=1)
parser.add_argument('--debug', dest='debug', action='store_true')
parser.set_defaults(debug = False)

args = parser.parse_args()
