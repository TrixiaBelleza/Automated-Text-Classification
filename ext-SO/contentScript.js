// var threshold = 50;

// chrome.storage.sync.get({
// 	threshold:'50'
// }, function(items) {
// 	threshold = items.threshold;
// });

//Deal with newly loaded tweets
// function DOMModificationHandler(){
// 	$(this).unbind('DOMSubtreeModified.event1');
// 	setTimeout(function() {
// 		modify();
// 		$('#timeline').bind('DOMSubtreeModified.event1', DOMModificationHandler);
// 	}, 10);
// }
// $('#timeline').bind('DOMSubtreeModified.event1', DOMModificationHandler);

//find and modify tall tweets
$('#wmd-input').each(function(index){
	var t = $(this);
	console.log(t.text())
	// var len = t.split(/\r\n|\r|\n/).length;

	// if(!$(this).hasClass("squished") && len > threshold){
	// 	$(this).addClass("squished");
	// 	$(this).html(`<button class="squish-button EdgeButton EdgeButton--primary" data-original-content="${encodeURI(t)}">Show Long Tweet</button>`);
		
	// 	//Add listeners
	// 	chrome.runtime.sendMessage({message:"listeners"}, function(response) {
	// 	});
	// }
});
