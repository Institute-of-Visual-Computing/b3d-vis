#pragma once
#include <owlViewer/OWLViewer.h>

class NanoViewer : public owl::viewer::OWLViewer
{
public:
	explicit NanoViewer(const std::string& title = "Sample Viewer", const int initWindowWidth = 1980,
						const int initWindowHeight = 1080)
		: owl::viewer::OWLViewer(title, owl::vec2i(initWindowWidth, initWindowHeight))
	{
	}
	auto showAndRunWithGui() -> void;
	auto showAndRunWithGui(const std::function<bool()>& keepgoing) -> void;

protected:
	virtual ~NanoViewer(){};
	virtual auto gui() -> void;
	virtual auto onFrameBegin() -> void
	{
	}
	virtual auto onFrameEnd() -> void
	{
	}

private:
	auto draw1() -> void;
};
