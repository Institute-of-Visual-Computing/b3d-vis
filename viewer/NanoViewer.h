#pragma once
#include <owlViewer/OWLViewer.h>

struct NanoViewer : public owl::viewer::OWLViewer
{
	NanoViewer(const std::string& title = "Sample Viewer", const int initWindowWidth = 1980,
	           const int initWindowHeight = 1080)
	    : owl::viewer::OWLViewer(title, owl::vec2i(initWindowWidth, initWindowHeight))
	{
	}

	virtual void gui();
	virtual void onFrameBegin()
	{
	}
	virtual void onFrameEnd()
	{
	}
	void showAndRunWithGui();
	void showAndRunWithGui(std::function<bool()> keepgoing);

  private:
	void initializeGui();
	void deinitializeGui();
};
