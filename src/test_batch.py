from od_api import ObjDetAPI
import sys
if __name__ == "__main__":
    save_path = None
    display = False
    display_num = 5
    num_box = 10
    thresh = 0.7
    if 'help' in sys.argv:
        print('Example call:\n \
            python test_batch.py -c ckpt-1 -i <image_path> -s <save_path> -d -dn 5 -n 10 -t 0.7\n \
            \n -c \t checkpoint flag (Required) \
            \n -i \t image directory path flag (Required) \
            \n -s \t save path flag, default: None (Optional) \
            \n -d \t display result flag, default: False (Optional) \
            \n -dn \t display num flag, default: 5 (Optional) \
            \n -n \t num boxes to draw flag, default: 10 (Optional) \
            \n -t \t detection threshold flag, default: 0.7 (Optional)')
        sys.exit()

    api = ObjDetAPI()
    if '-c' not in sys.argv:
        print('checkpoint not specified\ncall \'python test_batch.py help\' for more information')
        sys.exit()

    model = api.load_model(sys.argv[sys.argv.index('-c')+1])

    if '-i' not in sys.argv:
        print('image directory path not specified\ncall \'python test_batch.py help\' for more information')
        sys.exit()
    
    dir_path = sys.argv[sys.argv.index('-i')+1]

    if '-s' in sys.argv:
        save_path = sys.argv[sys.argv.index('-s')+1]
    
    if '-d' in sys.argv:
        display = True
    
    if '-n' in sys.argv:
        num_box = sys.argv[sys.argv.index('-n')+1]
    
    if '-t' in sys.argv:
        thresh = sys.argv[sys.argv.index('-t')+1]

    api.detect_batch(model, dir_path = dir_path, save_path= save_path, display = display,display_num = display_num,num_boxes = num_box, thresh = thresh)
    sys.exit()