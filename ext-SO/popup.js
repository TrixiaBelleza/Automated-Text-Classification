// //Saves options to chrome.storage
// function save_options() {
// 	var squishThreshold = document.getElementById('len').value;
// 	chrome.storage.sync.set({
// 		threshold: squishThreshold
// 	}, function() {
// 		// Update status to let user know options were saved
// 		var status = document.getElementById('status');
// 		status.textContent = 'Saved.';
// 		setTimeout(function() {
// 			status.textContent = '';
// 		}, 750);
// 	});
// }


// //Retrieves user settings
// function restore_options() {
// 	chrome.storage.sync.get({
// 		threshold: '50'
// 	}, function(items) {
// 		document.getElementById('len').value = items.threshold;
// 	});
// }

// $(document).ready(function() {
// 	restore_options();
// });
// document.getElementById('save').addEventListener('click', save_options);