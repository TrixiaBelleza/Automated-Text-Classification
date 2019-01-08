var tagrecom_div = document.createElement('div');
tagrecom_div.id = "tagrecom_div"
tagrecom_div.class = "tagrecom_div"
document.getElementById("tag-editor").appendChild(tagrecom_div);

var tagrecom_btn = document.createElement('a');
tagrecom_btn.id = "tagrecom_btn";
tagrecom_btn.classList.add("tagrecom_btn");
tagrecom_btn.appendChild(document.createTextNode("Tag Suggestions"));
tagrecom_btn.href = '#';
tagrecom_btn.id = "tagrecom_btn";
document.getElementById('tagrecom_div').appendChild(tagrecom_btn)

var backdrop_div = document.createElement('div');
backdrop_div.id = "backdrop_div";
backdrop_div.classList.add("backdrop_div");
document.getElementById('question-only-section').appendChild(backdrop_div);

var modal_content = document.createElement('div');
modal_content.id = "modal_content";

document.getElementById('backdrop_div').appendChild(modal_content);

var modal_body = document.createElement('h1');
modal_body.appendChild(document.createTextNode("Tag Suggestions"));
document.getElementById('modal_content').appendChild(modal_body);


var backdrop = document.querySelector('.backdrop_div');
var modal = document.querySelector('#modal_content');
console.log("modal:");
console.log(modal);
backdrop.style.display = 'none';
modal.style.display = 'none';

function openModal() {
	backdrop.style.display = 'block';
	modal.style.display = 'block';
	
}

// var close_btn = document.createElement('span');
// close_btn.class = "close_button";
// document.getElementById('modal_content').appendChild(close_btn);

// var modal_popup = document.querySelector(".modal");
// var trigger = document.querySelector(".tagrecom_btn");
// var close = document.querySelector(".close_button");

document.getElementById('tagrecom_btn').onclick = function(){
	$('#wmd-input').each(function(){
		var t = $(this).val();
		console.log("T:" + t);
		openModal();
	});
}
// chrome.runtime.sendMessage({message: "listeners"}, function(response) {
// });

