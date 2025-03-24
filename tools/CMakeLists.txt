add_subdirectory(colorMap)
add_subdirectory(header)
add_subdirectory(fits)
add_subdirectory(nano)
add_subdirectory(sofia)
add_subdirectory(sofiaNanoPipeline)
add_subdirectory(project)

if(BUILD_SERVER)
	add_subdirectory(projectServer)
endif()
add_subdirectory(projectServerClient)

add_subdirectory(nanoCutterParser)
add_subdirectory(runtimeDataset)

add_subdirectory(fitsCutterNanoVdbConverter)
