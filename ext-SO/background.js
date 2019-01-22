chrome.runtime.onMessage.addListener(
	function(request, sender, sendResponse) {
		console.log("Entered");
		if (request.message == "listeners") {
			chrome.tabs.executeScript(null, {file: "injectedScript.js"});	
		}
	}
);