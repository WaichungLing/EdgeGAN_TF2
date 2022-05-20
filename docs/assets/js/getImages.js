function onClick(arg) {
    var data_ = arg.getAttribute('data');
    var data_type = arg.getAttribute('data-type');
    var output = document.getElementById("output");
    var sketch_path = '';
    var gt_path = '';
    var json_path = '';
    var result_array = [];
    if (data_ == "Objects") {
        sketch_path = 'Object/Sketch/';
        gt_path = 'Object/GT/';
        json_path = 'https://cdn.jsdelivr.net/gh/sysu-imsl/CDN-for-SketchyCOCO@v1.0/Object.json'
        json_path = 'https://cdn.jsdelivr.net/gh/sysu-imsl/CDN-for-SketchyCOCO-added@1.5/Object_total.json'
    } else {
        sketch_path = 'Scene/Sketch/';
        gt_path = 'Scene/GT/';
        json_path = 'https://cdn.jsdelivr.net/gh/sysu-imsl/CDN-for-SketchyCOCO@v1.0/Scene.json'
    }
    $.ajax({
        type:'get',
        url:json_path,
        dataType:'json',
        success:function(data){
            console.log(data_type)
            if (data_ == "Objects") {
                if (data_type.toString() in data['added']) {
                    var tmp = data['added'][data_type.toString()];
                    var tmp1 = data['origin'][data_type.toString()];
                } else {
                    var tmp = data['origin'][data_type.toString()];
                    var tmp1 = 0;
                }

            } else {
                var tmp = data.data;
                var tmp1 = 0;
            }
            // console.log(tmp)
            var counter = 0;
            var index = 0;
            var names = new Array()
            if (tmp1 != 0) {
                var pre_head = "https://cdn.jsdelivr.net/gh/sysu-imsl/CDN-for-SketchyCOCO-added@1.5/";
            } else {
                var pre_head = "https://cdn.jsdelivr.net/gh/sysu-imsl/CDN-for-SketchyCOCO@v1.0/data/";
            }
            for (var n = 0; n<tmp.length; n++) {
                path = tmp[n];
                // name = path.split('/')[1].split('.')[0]
                // names[counter % 2] = name
                if (counter % 2 == 0 && counter != 0) {
                    result_array.push("        </tr>\n        <tr>\n          <td>\n            <a href=" + pre_head + sketch_path + path + ">" + "<img src=" + pre_head + sketch_path + path +  " alt=\"\" /></div></a>" + "\n       </td>");
                    result_array.push("          <td style=\"border-right: 3px double\">\n            <a href=" + pre_head + gt_path + path + ">" + "<img src=" + pre_head + gt_path + path +  " alt=\"\" /></div></a>" + "\n       </td>");
                } else {
                    result_array.push("          <td>\n            <a href=" + pre_head + sketch_path + path + ">" + "<img src=" + pre_head + sketch_path + path +  " alt=\"\" /></div></a>" + "\n       </td>");
                    result_array.push("          <td style=\"border-right: 3px double\">\n            <a href=" + pre_head + gt_path + path + ">" + "<img src=" + pre_head + gt_path + path +  " alt=\"\" /></div></a>" + "\n       </td>");
                    if (counter != 0) {
                        // result_array.push("        <tr>\n         <td>sketch(" + names[0] + ")</td>\n         <td style=\"border-right: 3px double\">ground-truth(" + names[0] + ")</td>         <td>sketch("+(names[1])+")</td>\n         <td>ground-truth("+(names[1])+")</td></tr>");
                        result_array.push("        <tr>\n         <td style=\"border-bottom: 3px double;\">sketch</td>\n         <td style=\"border-bottom: 3px double; border-right: 3px double\">ground truth</td>         <td  style=\"border-bottom: 3px double;\">sketch</td>\n         <td style=\"border-bottom: 3px double; border-right: 3px double\">ground-truth</td></tr>");
                        // result_array.push("        <tr>\n         <td style=\"border-right: 3px double\" colspan=\"2\">" + names[0] + "</td>         <td colspan=\"2\">"+(names[1])+"</td></tr>");
                        index = index + 2;
                    }
                }
                counter++;
            }

            if (tmp1 != 0) {
                var pre_head = "https://cdn.jsdelivr.net/gh/sysu-imsl/CDN-for-SketchyCOCO@v1.0/data/";
                for (var n = 0; n<tmp1.length; n++) {
                    path = tmp1[n];
                    // name = path.split('/')[1].split('.')[0]
                    // names[counter % 2] = name
                    if (counter % 2 == 0 && counter != 0) {
                        result_array.push("        </tr>\n        <tr>\n          <td>\n            <a href=" + pre_head + sketch_path + path + ">" + "<img src=" + pre_head + sketch_path + path +  " alt=\"\" /></div></a>" + "\n       </td>");
                        result_array.push("          <td style=\"border-right: 3px double\">\n            <a href=" + pre_head + gt_path + path + ">" + "<img src=" + pre_head + gt_path + path +  " alt=\"\" /></div></a>" + "\n       </td>");
                    } else {
                        result_array.push("          <td>\n            <a href=" + pre_head + sketch_path + path + ">" + "<img src=" + pre_head + sketch_path + path +  " alt=\"\" /></div></a>" + "\n       </td>");
                        result_array.push("          <td style=\"border-right: 3px double\">\n            <a href=" + pre_head + gt_path + path + ">" + "<img src=" + pre_head + gt_path + path +  " alt=\"\" /></div></a>" + "\n       </td>");
                        if (counter != 0) {
                            // result_array.push("        <tr>\n         <td>sketch(" + names[0] + ")</td>\n         <td style=\"border-right: 3px double\">ground-truth(" + names[0] + ")</td>         <td>sketch("+(names[1])+")</td>\n         <td>ground-truth("+(names[1])+")</td></tr>");
                            result_array.push("        <tr>\n         <td style=\"border-bottom: 3px double;\">sketch</td>\n         <td style=\"border-bottom: 3px double; border-right: 3px double\">ground truth</td>         <td  style=\"border-bottom: 3px double;\">sketch</td>\n         <td style=\"border-bottom: 3px double; border-right: 3px double\">ground-truth</td></tr>");
                            // result_array.push("        <tr>\n         <td style=\"border-right: 3px double\" colspan=\"2\">" + names[0] + "</td>         <td colspan=\"2\">"+(names[1])+"</td></tr>");
                            index = index + 2;
                        }
                    }
                    counter++;
                }
            }
            if (counter % 2 != 0) {
                result_array.push("        <tr>\n         <td  style=\"border-bottom: 3px double;\">sketch</td>\n         <td style=\"border-bottom: 3px double; border-right: 3px double\">ground truth</td>   </tr>");
            }
            console.log(counter)
            output.innerHTML = "              <table>\n        <tr>\n" + result_array.join("\n") + "        </tr>\n      </table>";
        }
    })

}


$(document).ready(function () {
    // document.getElementById("first").click();
    $('#start').trigger("onclick")
});

// ele_1009
// ele_67
// ele_1030
// ele_1179
// ele_1080
// ele_1052
// ele_26
// ele_1174
// ele_1029
// ele_758
// ele_62