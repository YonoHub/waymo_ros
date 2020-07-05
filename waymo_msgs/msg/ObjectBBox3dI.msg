uint32 id                               # The id of the object in the frame

float64 x                               # The relative x position of the center of object
float64 y                               # The relative y position of the center of object
float64 z                               # The relative z position of the center of object

float64 width                           # The width of the bbox
float64 length                          # The length of the bbox
float64 height                          # The height of the bbox
float64 heading                         # The heading of the bbox

float64 speed_x                         # The speed of the object in x direction
float64 speed_y                         # The speed of the object in y direction

float64 acc_x                           # The acceleration of the object in x direction
float64 acc_y                           # The acceleration of the object in y direction
    
LabelW[] labels                         # The most probable labels of the object
                                        # ordered descendingly according to their confidence