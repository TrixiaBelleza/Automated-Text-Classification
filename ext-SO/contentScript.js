var tagrecom_div = document.createElement('div');
tagrecom_div.id = "tagrecom_div";
document.getElementById("tag-editor").appendChild(tagrecom_div);

var tagrecom_btn = document.createElement('a');
tagrecom_btn.id = "tagrecom_btn";
tagrecom_btn.appendChild(document.createTextNode("Tag Suggestions"));
document.getElementById('tagrecom_div').appendChild(tagrecom_btn)

var modal_content = document.createElement('div');
modal_content.id = "modal_content";
document.getElementById('tagrecom_div').appendChild(modal_content);

var close_btn = document.createElement('a');
close_btn.id = "close_btn";
close_btn.appendChild(document.createTextNode("CLOSE"));
document.getElementById('tagrecom_div').appendChild(close_btn);
close_btn.style.display = 'none';

var modal = document.querySelector('#modal_content');
modal.style.display = 'none';

var count = 0;
function openModal() {
	count = count + 1;
	modal.style.display = 'block';
	close_btn.style.display = 'block';
	if(count == 1){
		$.ajax({
		    url: "https://9f638471.ngrok.io/_get_text_input/",
		    data: $('#wmd-input').val(),
		    type: "POST",
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

}

function closeModal() {
	modal.style.display = 'none';
	close_btn.style.display = 'none';
}

$(document).ready(function() {
	document.getElementById('tagrecom_btn').onclick = function(){
		$('#wmd-input').each(function(){
			console.log($(this).val())
			openModal();
		});	
	}

	document.getElementById('close_btn').onclick = function() {
		closeModal();
	}
	
});