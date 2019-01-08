var tagrecom_div = document.createElement('div');
tagrecom_div.id = "tagrecom_div";
document.getElementById("tag-editor").appendChild(tagrecom_div);

var tagrecom_btn = document.createElement('a');
tagrecom_btn.id = "tagrecom_btn";
tagrecom_btn.appendChild(document.createTextNode("Tag Suggestions"));
document.getElementById('tagrecom_div').appendChild(tagrecom_btn)

var modal_content = document.createElement('div');
modal_content.id = "modal_content";

document.getElementById('question-only-section').appendChild(modal_content);

var modal_body = document.createElement('h1');
modal_body.appendChild(document.createTextNode("Tag Suggestions"));
document.getElementById('modal_content').appendChild(modal_body);

var close_btn = document.createElement('a');
close_btn.id = "close_btn";
close_btn.appendChild(document.createTextNode("CLOSE"));
document.getElementById('modal_content').appendChild(close_btn);


var modal = document.querySelector('#modal_content');
modal.style.display = 'none';

function openModal() {
	modal.style.display = 'block';
}

function closeModal() {
	modal.style.display = 'none';
}

document.getElementById('tagrecom_btn').onclick = function(){
	$('#wmd-input').each(function(){
		// var t = $(this).val();
		// console.log("T:" + t);
		openModal();
	});
}

document.getElementById('close_btn').onclick = function() {
	closeModal();
}
// chrome.runtime.sendMessage({message: "listeners"}, function(response) {
// });

