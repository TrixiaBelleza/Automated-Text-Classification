chrome.runtime.onMessage.addListener(
	function(request, sender, sendResponse) {
		if (request.message == "listeners") {
			chrome.tabs.executeScript(null, {file: "injectedScript.js"});
			sendResponse({message: "OK"});	
		}
	}
);