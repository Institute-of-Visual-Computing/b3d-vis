#include "GuiDefaultLayoutSettings.h"

std::string_view defaultLayoutSettings =
	R"([Window][DockSpaceViewport_11111111]
Pos=0,30
Size=1980,1050
Collapsed=0

[Window][Volume Viewport##3499211612]
Pos=0,30
Size=990,1050
Collapsed=0
DockId=0x00000001,0

[Window][Transfer Mapping##581869302]
Pos=992,30
Size=988,516
Collapsed=0
DockId=0x00000009,0

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
DockId=0x00000005,0

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

[Window][SoFiA-Search##3890346734]
Pos=992,548
Size=988,312
Collapsed=0
DockId=0x0000000A,0

[Window][Project##3586334585]
Pos=992,862
Size=988,218
Collapsed=0
DockId=0x00000008,0

[Window][###modal4161255391]
Pos=495,291
Size=976,498
Collapsed=0

[Window][###modal3922919429]
Pos=671,423
Size=974,414
Collapsed=0

[Window][###modal949333985]
Pos=510,451
Size=960,178
Collapsed=0

[Window][###modal545404204]
Pos=1193,474
Size=295,188
Collapsed=0

[Window][###modal2715962298]
Pos=510,451
Size=960,178
Collapsed=0

[Window][Example: Simple layout]
Pos=60,60
Size=500,440
Collapsed=0

[Window][Example: Simple layout/left pane_AED60EF8]
IsChild=1
Size=150,364

[Window][WindowOverViewport_11111111]
Pos=0,30
Size=1980,1050
Collapsed=0

[Table][0xD181190E,2]
Column 0  Weight=1.0000
Column 1  Weight=1.0000

[Table][0x59BB87A5,4]
RefScale=24
Column 0  Width=394
Column 1  Width=78
Column 2  Width=70
Column 3  Width=128

[Docking][Data]
DockSpace       ID=0x7C6B3D9B Window=0xA87D555D Pos=852,490 Size=1980,1050 Split=X Selected=0x264F2F11
  DockNode      ID=0x00000001 Parent=0x7C6B3D9B SizeRef=990,1050 CentralNode=1 HiddenTabBar=1 Selected=0x264F2F11
  DockNode      ID=0x00000006 Parent=0x7C6B3D9B SizeRef=988,1050 Split=Y Selected=0x7D4F20AD
    DockNode    ID=0x00000007 Parent=0x00000006 SizeRef=988,830 Split=Y Selected=0x7D4F20AD
      DockNode  ID=0x00000009 Parent=0x00000007 SizeRef=988,516 Selected=0x7D4F20AD
      DockNode  ID=0x0000000A Parent=0x00000007 SizeRef=988,312 Selected=0xFB2E6F72
    DockNode    ID=0x00000008 Parent=0x00000006 SizeRef=988,218 Selected=0x552EC863
DockSpace       ID=0x8B93E3BD Pos=429,493 Size=1980,1050 Split=X Selected=0x264F2F11
  DockNode      ID=0x00000004 Parent=0x8B93E3BD SizeRef=1039,1050 Split=Y
    DockNode    ID=0x00000002 Parent=0x00000004 SizeRef=1980,786 CentralNode=1 HiddenTabBar=1 Selected=0x264F2F11
    DockNode    ID=0x00000003 Parent=0x00000004 SizeRef=1980,262 Selected=0xC5A34CEF
  DockNode      ID=0x00000005 Parent=0x8B93E3BD SizeRef=939,1050 Selected=0xE87781F4


)";
