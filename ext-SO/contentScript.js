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
	    url: "https://e69d2257.ngrok.io/_get_text_input/",
	    data: $('#wmd-input').val(),
	    type: "POST",
	    contentType : "application/json",
	    success: function(resp){
    	  $('#modal_content').append(resp.data);
    	  // var tags = "";
    	  // for(i=0; i<resp.predicted_tags.length; i++){
    	  // 	tags += resp.predicted_tags[i] + " ";
    	  // }
    	  var tagspan = document.createElement('span');
    	  tagspan.classList.add('s-tag', 'rendered-element');
    	  tagspan.appendChild(document.createTextNode("javascript"));
    	  document.getElementsByClassName('tag-editor s-input')[0].firstChild.appendChild(tagspan);

    	  var removeTag = document.createElement('a');
    	  removeTag.classList.add('js-delete-tag', 's-tag--dismiss');
    	  removeTag.setAttribute("title", "Remove tag");
    	  document.getElementsByClassName('s-tag rendered-element')[0].appendChild(removeTag);

    	  var removeTagSVG = document.createElement('svg');
    	  removeTagSVG.style.pointerEvents = 'none';
    	  removeTagSVG.classList.add('svg-icon', 'iconClearSm');
    	  removeTagSVG.setAttribute("height", "12");
    	  removeTagSVG.setAttribute("width", "12"); 
    	  removeTagSVG.setAttribute("viewBox", "0 0 14 14");
    	  removeTag.appendChild(removeTagSVG);

    	  var pathSVG = document.createElement('path');
    	  pathSVG.setAttribute("d", "M12 3.41L10.59 2 7 5.59 3.41 2 2 3.41 5.59 7 2 10.59 3.41 12 7 8.41 10.59 12 12 10.59 8.41 7z");
    	  removeTagSVG.appendChild(pathSVG);

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
