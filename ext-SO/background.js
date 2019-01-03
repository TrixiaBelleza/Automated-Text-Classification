// chrome.runtime.onMessage.addListener(
// 	function(request, sender, sendResponse) {
// 		if (request.message == "listeners"){
// 			//add event handler for button click
// 			chrome.tabs.executeScript(null, {file: "injectedScript.js"});
// 			sendResponse({message: "OK"});
// 		}
// 	}
// )