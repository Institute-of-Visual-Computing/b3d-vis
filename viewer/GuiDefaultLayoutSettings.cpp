#include "GuiDefaultLayoutSettings.h"

std::string_view defaultLayoutSettings =
	R"([Window][DockSpaceViewport_11111111]
Pos=0,30
Size=1980,1050
Collapsed=0

[Window][Volume Viewport##3499211612]
Pos=0,30
Size=1039,786
Collapsed=0
DockId=0x00000002,0

[Window][Transfer Mapping##581869302]
Pos=1041,557
Size=939,523
Collapsed=0
DockId=0x00000007,0

[Window][Debug##Default]
Pos=60,60
Size=400,400
Collapsed=0

[Window][[DEPRECATED] RT Settings]
ViewportPos=1261,1733
ViewportId=0x2DF39E99
Size=373,720
Collapsed=0

[Window][Dear ImGui Demo]
Pos=1041,30
Size=939,525
Collapsed=0
DockId=0x00000006,0

[Window][[DEPRECATED] Features]
ViewportPos=982,43
ViewportId=0x1B854E9A
Size=266,1311
Collapsed=0

[Window][###modal3890346734]
Pos=919,488
Size=128,104
Collapsed=0

[Window][###modal3586334585]
Pos=503,333
Size=974,414
Collapsed=0

[Window][###ProfilerWindow]
Pos=0,818
Size=1039,262
Collapsed=0
DockId=0x00000003,0

[Window][Example: Custom rendering]
ViewportPos=326,762
ViewportId=0x3AC84485
Size=3231,1680
Collapsed=0

[Window][Example: Property editor]
Pos=60,60
Size=430,450
Collapsed=0

[Window][Dear ImGui Metrics/Debugger]
Pos=1559,313
Size=1824,1725
Collapsed=0

[Table][0xD181190E,2]
Column 0  Weight=1.0000
Column 1  Weight=1.0000

[Docking][Data]
DockSpace     ID=0x8B93E3BD Window=0xA787BDB4 Pos=429,493 Size=1980,1050 Split=X Selected=0x264F2F11
  DockNode    ID=0x00000004 Parent=0x8B93E3BD SizeRef=1039,1050 Split=Y
    DockNode  ID=0x00000002 Parent=0x00000004 SizeRef=1980,786 CentralNode=1 HiddenTabBar=1 Selected=0x264F2F11
    DockNode  ID=0x00000003 Parent=0x00000004 SizeRef=1980,262 Selected=0xC5A34CEF
  DockNode    ID=0x00000005 Parent=0x8B93E3BD SizeRef=939,1050 Split=Y Selected=0xE87781F4
    DockNode  ID=0x00000006 Parent=0x00000005 SizeRef=939,525 Selected=0xE87781F4
    DockNode  ID=0x00000007 Parent=0x00000005 SizeRef=939,523 Selected=0x7D4F20AD

)";
