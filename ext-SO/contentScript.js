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
	$.ajax({
			url: "https://3cf785c6.ngrok.io/_fetch_input/",
			data: $('#wmd-input').val(),
			type: 'POST',
			success: function(response){
				console.log(response);
			},
			error: function(error){
				console.log(error);
			}
		});
	$.ajax({
	    url: "https://3cf785c6.ngrok.io/_get_data/",
	    type: "GET",
	    contentType : "application/json",
	    success: function(resp){
	      console.log(resp);
	      $('#modal_content').append(resp.data);
	    },
	    error: function(e, s, t) {
	      console.log("ERROR OCCURRED");
	      console.log(e);
	      console.log(s);
	      console.log(t);
		}
	});

}

function closeModal() {
	modal.style.display = 'none';
}
$(document).ready(function() {
	document.getElementById('tagrecom_btn').onclick = function(){
		$('#wmd-input').each(function(){
		
			console.log($(this).val())
			openModal();
			// chrome.runtime.sendMessage({message: "listeners"}, function(response) {
			// // });



		});
		
	}

	document.getElementById('close_btn').onclick = function() {
		closeModal();
	}
});