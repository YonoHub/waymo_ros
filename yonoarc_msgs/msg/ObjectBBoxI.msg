uint32 id                               # The id of the object in the frame
uint32 parent_id                        # The id of the parent object in the frame

float64 x                               # The x coordinate of the top-left corner of the bbox
float64 y                               # The y coordinate of the top-left corner of the bbox
float64 width                           # The width of the bbox
float64 height                          # The height of the bbox

LabelI[] labels                         # The most probable labels of the object
                                        # ordered descendingly according to their confidence