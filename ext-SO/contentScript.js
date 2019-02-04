var tagrecom_div = document.createElement('div');
tagrecom_div.id = "tagrecom_div";
document.getElementById("tag-editor").appendChild(tagrecom_div);

var tagrecom_btn = document.createElement('a');
tagrecom_btn.id = "tagrecom_btn";
tagrecom_btn.appendChild(document.createTextNode("Tag Suggestions"));
tagrecom_btn.style.fontSize = 'medium';
document.getElementById('tagrecom_div').appendChild(tagrecom_btn)

var modal_content = document.createElement('div');
modal_content.id = "modal_content";
document.getElementById('tagrecom_div').appendChild(modal_content);

var modal = document.querySelector('#modal_content');
modal.style.display = 'none';

var loader = document.createElement('img');
loader.src = chrome.runtime.getURL("simple-loader.gif");
loader.style.width = '30px';
loader.style.height = '30px';
document.getElementById('tagrecom_div').appendChild(loader);
loader.style.display = 'none';

var close_btn = document.createElement('a');
close_btn.id = "close_btn";
close_btn.appendChild(document.createTextNode("CLOSE"));
document.getElementById('tagrecom_div').appendChild(close_btn);
close_btn.style.display = 'none';

function openModal() {
	modal.style.display = 'block';
	close_btn.style.display = 'block';
	loader.style.display = 'block';
	$.ajax({
	    url: "https://9f638471.ngrok.io/_get_text_input/",
	    data: $('#wmd-input').val(),
	    type: "POST",
	    contentType : "application/json",
	    success: function(resp){
    	  $('#modal_content').append(resp.data);
		  loader.style.display = 'none';
	    },
	    error: function(e, s, t) {
	      console.log(e);
	      console.log(s);
	      console.log(t);
		}
	});
}

function closeModal() {
	modal.style.display = 'none';
	close_btn.style.display = 'none';
}

$(document).ready(function() {
	document.getElementById('tagrecom_btn').onclick = function(){

		while (modal_content.firstChild) {
    		modal_content.removeChild(modal_content.firstChild);
		}
		$('#wmd-input').each(function(){
			openModal();
		});	
	}

	document.getElementById('close_btn').onclick = function() {
		closeModal();
	}
	
});