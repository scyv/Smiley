/* global $ */
class Main {
  constructor() {
    this.canvas = $("#main")[0];
    this.ctx = this.canvas.getContext("2d");
    this.input = $("#input")[0];

    this.numAugm = param.numAugm;
    this.batchSize = param.batchSize;
    this.cnnRate = param.cnnRate;
    this.cnnEpochs = param.cnnEpochs;
    this.image_size = param.image_size;
    this.rect_size = 448; // 16 * 28
    this.col_width = this.rect_size / this.image_size; // for the grid

    this.canvas.width = this.rect_size;
    this.canvas.height = this.rect_size;
    this.input.width = 5 * this.image_size;
    this.input.height = 5 * this.image_size;

    var catsList = $("#trainingDataLabelOptions")[0];
    this.cats = param.categories;
    this.cats.forEach(function (item) {
      $("<option></option>").val(item).appendTo(catsList);
    });
    this.fixed_cats = this.cats.filter(
      (x) => !param.user_categories.includes(x)
    );
    this.cats_img_number = param.cats_img_number;
    this.maxNumUserCat = param.maxNumUserCat;

    this.canvas.addEventListener("mousedown", this.onMouseDown.bind(this));
    this.canvas.addEventListener("mouseup", this.onMouseUp.bind(this));
    this.canvas.addEventListener("mousemove", this.onMouseMove.bind(this));
    this.canvas.addEventListener("mouseout", this.onMouseOut.bind(this));

    this.createCategoryButtons();
    this.initializeConfigValues();
    this.initialize();
  }

  initialize() {
    this.clearOutput();

    if (this.video) {
      this.video.play();

      this.makeMenuActive($("#modeCamera"));
      $("#takePhoto").show();
    } else {
      this.ctx.fillStyle = "#FFFFFF";
      this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
      this.input
        .getContext("2d")
        .clearRect(0, 0, this.input.width, this.input.height);

      this.ctx.lineWidth = 0.05;
      for (var i = 0; i < this.image_size; i++) {
        this.ctx.beginPath();
        this.ctx.moveTo((i + 1) * this.col_width, 0);
        this.ctx.lineTo((i + 1) * this.col_width, this.rect_size);
        this.ctx.closePath();
        this.ctx.stroke();

        this.ctx.beginPath();
        this.ctx.moveTo(0, (i + 1) * this.col_width);
        this.ctx.lineTo(this.rect_size, (i + 1) * this.col_width);
        this.ctx.closePath();
        this.ctx.stroke();
      }

      this.makeMenuActive($("#modeDraw"));
      $("#takePhoto").hide();
    }
  }

  initializeConfigValues() {
    $("#num-augm").val(this.numAugm);
    $("#batch-size").val(this.batchSize);
    $("#cnn-rate").val(this.cnnRate);
    $("#cnn-epochs").val(this.cnnEpochs);
  }

  createCategoryButtons() {
    var fixed_cat_div = $("#categories .fixed-categories");
    var user_cat_div = $("#categories .user-categories");
    const sort_fn = (a, b) => {
      return $(a).text() > $(b).text() ? 1 : -1;
    };

    this.fixed_cats.forEach((item) => {
      this.addCategoryButton(item, fixed_cat_div, true);
    });
    $(fixed_cat_div).children().sort(sort_fn).appendTo(fixed_cat_div);

    this.cats.forEach((item) => {
      if (!this.fixed_cats.includes(item)) {
        this.addCategoryButton(item, user_cat_div, false);
      }
    });
    $(user_cat_div).children().sort(sort_fn).appendTo(user_cat_div);

    this.addNumberToCategories();
  }

  addCategoryButton(category, location, fixed) {
    var outerDiv = document.createElement("div");
    $(outerDiv).addClass("input-group col-sm-6");

    var button = document.createElement("div");
    $(button)
      .addClass("btn btn-outline-secondary rounded")
      .html(category)
      .val(category)
      .click((e) => {
        this.addTrainingData(outerDiv, $(e.currentTarget).val());
      })
      .appendTo(outerDiv);

    if (fixed) {
      $(button).addClass("button-own-image " + category + "-img");
    }

    location.append(outerDiv);
  }

  addNumberToCategories() {
    for (var key in this.cats_img_number) {
      // check if the property/key is defined in the object itself, not in parent
      if (this.cats_img_number.hasOwnProperty(key)) {
        this.updateCategoryButton(key, this.cats_img_number[key]);
      }
    }
  }

  /*
        Parameter:
            - category: category name
            - number: number of images for category, or <0 if current number should be increased by one
    */
  updateCategoryButton(category, number) {
    var button = $("#categories .btn").filter(function () {
      return this.value == category;
    })[0];

    var value = $(button).children().text().replace(/\D/g, "");
    $(button).children().remove();

    var numberDiv = document.createElement("div");
    numberDiv.id = category + "-number";
    $(numberDiv)
      .html(" (" + (number >= 0 ? number : Number(value) + 1) + ")")
      .addClass("inline")
      .appendTo(button);

    if (!value) {
      this.addFolderToCategory(category, $(button).parent());
      this.addDeleteToCategory(category, $(button).parent());
    }
  }

  addFolderToCategory(category, location) {
    $(location).children().removeClass("rounded");

    var button = document.createElement("div");
    $(button)
      .addClass("input-group-append btn btn-outline-secondary")
      .html("<i class='fa fa-folder-open'></i>")
      .click((e) => {
        this.open_category_folder(category);
        e.stopPropagation();
      })
      .appendTo(location);
  }

  addDeleteToCategory(category, location) {
    $(location).children().removeClass("rounded");

    var button = document.createElement("div");
    $(button)
      .addClass("input-group-append btn btn-outline-danger")
      .html("<i class='fa fa-times'></i>")
      .click((e) => {
        this.deleteCategory(category, location);
        e.stopPropagation();
      })
      .appendTo(location);
  }

  deleteCategory(category, button) {
    $.ajax({
      url: "/api/delete-category",
      method: "POST",
      contentType: "application/json",
      data: JSON.stringify({ cat: category }),
      success: (data) => {
        $(
          "#trainingDataLabelOptions option[value='" + category + "']"
        ).remove(); // delete cat from datalist options
        $(button).find('div[id$="number"]').remove();
        $(button)
          .find(".button-own-image")
          .addClass("rounded")
          .siblings()
          .remove();

        if (!this.fixed_cats.includes(category)) {
          $(button).remove();

          var index = this.cats.indexOf(category);
          if (index !== -1) {
            this.cats.splice(index, 1);
          }
        }

        this.initialize();
      },
    }).fail(() => {
      this.clearOutput();
      this.checkConnection();
    });
  }

  onMouseDown(e) {
    if (this.video) return; // don't draw in camera mode

    this.prev = this.getPosition(e.clientX, e.clientY);
    this.drawing = true;
  }

  onMouseUp(e) {
    if (this.video) this.video.pause();

    this.onMouseMove(e);
    this.recogniseInput((input) => {});
    this.drawing = false;
  }

  onMouseOut() {
    if (this.drawing) this.recogniseInput((input) => {});

    this.drawing = false;
  }

  onMouseMove(e) {
    if (this.drawing) {
      var curr = this.getPosition(e.clientX, e.clientY);
      this.ctx.lineWidth = Math.max(5, 46 - this.image_size / 2);
      this.ctx.lineCap = "round";
      this.ctx.beginPath();
      this.ctx.moveTo(this.prev.x, this.prev.y);
      this.ctx.lineTo(curr.x, curr.y);
      this.ctx.stroke();
      this.ctx.closePath();
      this.prev = curr;
    }
  }

  getPosition(clientX, clientY) {
    var rect = this.canvas.getBoundingClientRect();
    return {
      x: clientX - rect.left,
      y: clientY - rect.top,
    };
  }

  drawInput(cb) {
    var ctx = this.input.getContext("2d");
    var img = new Image();
    img.onload = () => {
      var input = [];
      var small = document.createElement("canvas").getContext("2d");
      small.drawImage(
        img,
        0,
        0,
        img.width,
        img.height,
        0,
        0,
        this.image_size,
        this.image_size
      );
      var data = small.getImageData(
        0,
        0,
        this.image_size,
        this.image_size
      ).data;

      // get max and min for normalization
      var max = 0;
      var min = 0;
      for (var i = 0; i < this.image_size; i++) {
        for (var j = 0; j < this.image_size; j++) {
          var n = 4 * (i * this.image_size + j);
          var grayscale = (data[n + 0] + data[n + 1] + data[n + 2]) / 3;
          max = Math.max(max, grayscale);
          min = Math.min(min, grayscale);
        }
      }

      for (var i = 0; i < this.image_size; i++) {
        for (var j = 0; j < this.image_size; j++) {
          var n = 4 * (i * this.image_size + j);
          var grayscale = (data[n + 0] + data[n + 1] + data[n + 2]) / 3;
          grayscale = (255 * (grayscale - min)) / (max - min);

          // Threshold
          const threshold = 80;
          const contrast_factor = 2;
          var scaled_gray = Math.min(
            255,
            (grayscale - threshold) * contrast_factor + threshold
          );
          grayscale = grayscale > threshold ? scaled_gray : grayscale;

          input[i * this.image_size + j] = grayscale;
          ctx.fillStyle = "rgb(" + Array(3).fill(grayscale) + ")";
          ctx.fillRect(j * 5, i * 5, 5, 5);
        }
      }
      if (Math.min(...input) === 255) {
        $(this.canvas).addClass("red-box-shadow");
        setTimeout(() => {
          $(this.canvas).removeClass("red-box-shadow");
        }, 1000);
        return;
      }
      cb(input);
    };
    img.src = this.canvas.toDataURL();
  }

  clearOutput() {
    this.dismissAlert();
    $("#output td, #output tr").remove();
  }

  dismissAlert() {
    $("#error")
      .removeClass((i, className) => {
        return (className.match(/(^|\s)alert-\S+/g) || []).join(" ");
      })
      .hide()
      .find("p")
      .text("");
    clearTimeout(this.alert_timeout);
  }

  displayAlert(content, type) {
    var alert = $("#error");
    alert
      .addClass("alert-" + type)
      .fadeIn()
      .find("p")
      .html(content.replace(/(\r\n|\n|\r)/gm, "<br>"));
    this.alert_timeout = setTimeout(() => {
      alert.fadeOut();
    }, 60000);
  }

  recogniseInput(cb) {
    if (this.video) this.video.pause();

    this.drawInput((input) => {
      typeof cb == "function" ? cb(input) : this.loadOutput(input);
    });
  }

  loadOutput(input) {
    $.ajax({
      url: "/api/classify",
      method: "POST",
      contentType: "application/json",
      data: JSON.stringify(input),
      success: (data) => {
        const error = data.error;
        var categories = data.categories;
        var classifiers = data.classifiers;
        var results = data.results;

        if (error) {
          this.displayAlert(error, "warning");
        }

        // Do not display table if results contain empty arrays
        if (
          !results.filter((e) => {
            return e.length;
          }).length
        )
          return;
        else {
          // Display user categories as last
          const rm_duplicates = (v, i, a) => a.lastIndexOf(v) == i;

          categories.forEach((v, i) => {
            if (!this.fixed_cats.includes(v)) {
              for (var j in results) results[j].push(results[j][i]);
              categories.push(v);
            }
          });

          categories = categories.filter(rm_duplicates);
          for (var j in results) results[j] = results[j].filter(rm_duplicates);
        }

        const table = $("#output");
        const thead = $("<thead>");
        const tbody = $("<tbody>");
        table.empty();
        table.append(thead);
        table.append(tbody);

        const headRow = $("<tr>");
        thead.append(headRow);
        headRow.append("<th>Network Output</th>");
        for (
          let classifierIdx = 0;
          classifierIdx < classifiers.length;
          classifierIdx++
        ) {
          const cell = $("<th scope='col' class='text-right'>");
          headRow.append(cell);
          cell.text(classifiers[classifierIdx]);
        }

        const mostSuccessfulCells = [];
        const mostSuccessfulValues = [];

        for (
          let categoryIdx = 0;
          categoryIdx < categories.length;
          categoryIdx++
        ) {
          const row = $("<tr>");
          tbody.append(row);
          const categoryNameCell = $("<th scope='row'>");

          const outerDiv = document.createElement("div");
          $(outerDiv).addClass("input-group");
          const textElement = $(
            "<span class='button-own-image " +
              categories[categoryIdx] +
              "-img'>"
          );
          textElement.text(categories[categoryIdx]).appendTo(outerDiv);
          categoryNameCell.append(outerDiv);
          row.append(categoryNameCell);

          for (
            let classifierIdx = 0;
            classifierIdx < classifiers.length;
            classifierIdx++
          ) {
            const cell = $("<td>");
            row.append(cell);
            const result = results[classifierIdx][categoryIdx];
            cell.text((result * 100).toFixed(2) + "%");
            const mostSuccessfulValue = mostSuccessfulValues[classifierIdx];
            if (!mostSuccessfulValue || result > mostSuccessfulValue) {
              mostSuccessfulValues[classifierIdx] = result;
              mostSuccessfulCells[classifierIdx] = cell;
            }
          }
        }

        for (let index = 0; index < mostSuccessfulCells.length; index++) {
          mostSuccessfulCells[index].addClass("table-success");
        }
      },
    }).fail(() => {
      this.clearOutput();
      this.checkConnection();
    });
  }

  open_category_folder(category) {
    $.ajax({
      url: "/api/open-category-folder",
      method: "POST",
      contentType: "application/json",
      data: JSON.stringify({ cat: category }),
    }).fail(() => {
      this.clearOutput();
      this.checkConnection();
    });
  }

  addTrainingData(button, label) {
    if (!label) alert("Please assign a category for the image");
    else if (
      this.cats.filter((e) => !this.fixed_cats.includes(e)).length ==
      this.maxNumUserCat
    )
      alert(
        "Maximum number of user-defined categories (" +
          this.maxNumUserCat +
          ") already reached"
      );
    else if (label.length > 12)
      alert("Maximum number of characters (12) exceeded");
    else if (!/^[a-zA-Z0-9\-]*$/.test(label))
      alert(
        "Please use only latin alphabet (a-z/A-Z), hyphen(-) and digits(0-9) for the category name"
      );
    else {
      this.recogniseInput((input) => {
        const uploadData = {
          cat: label,
          img: input,
        };

        $(button).fadeOut(400).fadeIn(400);
        var blink = setInterval(function () {
          $(button).fadeOut(400).fadeIn(400);
        }, 1000);

        this.uploadTrainingData(uploadData, blink);
      });
    }
  }

  uploadTrainingData(input, blink) {
    $.ajax({
      url: "/api/add-training-example",
      method: "POST",
      contentType: "application/json",
      data: JSON.stringify(input),
      success: (data) => {
        this.initialize();
        const error = data.error;
        if (error) {
          this.displayAlert(error, "warning");
        }

        var label = input.cat;
        if (!this.cats.includes(label)) {
          this.cats.push(label);
          var catsList = $("#trainingDataLabelOptions")[0];
          var option = document.createElement("option");
          $(option).val(label);
          catsList.append(option);

          this.addCategoryButton(label, $("#categories .user-categories")[0]);
        }

        this.updateCategoryButton(label, -1);
      },
    })
      .always(() => {
        clearInterval(blink);
      })
      .fail(() => {
        this.clearOutput();
        this.checkConnection();
      });
  }

  useModeDraw(button) {
    if (this.video) {
      this.video.pause();
      this.video.srcObject.getVideoTracks().forEach(function (track) {
        track.stop();
      });
      this.video = null;
    }

    this.initialize();
  }

  useModeCamera(button) {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      if (!this.video) {
        var constraints = {
          video: {
            width: this.rect_size,
            height: this.rect_size,
            facingMode: "user",
            frameRate: 10,
          },
        };

        navigator.mediaDevices
          .enumerateDevices()
          .then((deviceInfos) => {
            /* sets video_device_id to the last webcam found (edge-compatible)
                       or to the logitech one (chrome-compatible)*/
            for (var i in deviceInfos) {
              if (deviceInfos[i].kind === "videoinput")
                this.video_device_id = deviceInfos[i].deviceId;

              if (
                deviceInfos[i].label.startsWith("Logitech") &&
                deviceInfos[i].kind === "videoinput"
              )
                break;
            }

            constraints["video"]["deviceId"] = { exact: this.video_device_id };
            return navigator.mediaDevices.getUserMedia(constraints);
          })
          .then((mediaStream) => {
            this.drawing = false;

            const ctx = this.ctx;
            const rect_size = this.rect_size;

            this.video = document.createElement("video");
            this.video.srcObject = mediaStream;
            this.video.addEventListener(
              "play",
              function () {
                var $this = this;
                (function loop() {
                  if (!$this.paused && !$this.ended) {
                    ctx.drawImage($this, 0, 0, rect_size, rect_size);
                    setTimeout(loop, 1000 / constraints["video"]["frameRate"]); // drawing at 10fps
                  }
                })();
              },
              0
            );
            this.initialize();
          })
          .catch(function (err) {
            console.log(err.name + ": " + err.message);
          }); // always check for errors at the end.
      } else {
        this.initialize();
      }
    } else {
      alert("getUserMedia() is not supported by your browser");
    }
  }

  makeMenuActive(button) {
    $(button).addClass("menu-active");
    $(button).siblings().removeClass("menu-active");
  }

  trainModels(button) {
    if (this.is_training) {
      $(button).prop("disabled", true);
      $.ajax({
        url: "/api/stop-training",
        method: "POST",
      }).fail(() => {
        $(button).prop("disabled", false);
        this.clearOutput();
        this.checkConnection();
      });
    } else {
      this.is_training = true;

      $(button).children(".progress-bar").addClass("bg-danger");
      $(button)
        .children(".label-progress-bar")
        .text("Stop Training")
        .css("color", "black");

      var update_progress = setInterval(function () {
        $.ajax({
          url: "/api/train-progress",
          success: (data) => {
            $(button)
              .children(".progress-bar")
              .css("width", data.progress + "%");
          },
        });
      }, 800);

      $.ajax({
        url: "/api/train-models",
        method: "POST",
        success: (data) => {
          this.clearOutput();

          const error = data.error;
          if (error) {
            this.displayAlert(error, "warning");
          }
        },
      })
        .always(() => {
          clearInterval(update_progress);

          this.is_training = false;

          $(button).prop("disabled", false);
          $(button)
            .children(".progress-bar")
            .removeClass("bg-danger")
            .css("width", "100%");
          $(button)
            .children(".label-progress-bar")
            .text("Start Training")
            .css("color", "white");
        })
        .fail(() => {
          this.clearOutput();
          this.checkConnection();
        });
    }
  }

  updateConfig(form) {
    var ints = {
      numAugm: $("#num-augm").val(),
      batchSize: $("#batch-size").val(),
      cnnEpochs: $("#cnn-epochs").val(),
    };
    var floats = { cnnRate: $("#cnn-rate").val() };

    for (var i in ints) {
      if (!/^\+?\d+$/.test(ints[i])) {
        $(form)
          .find("i.fa-spinner.fa-spin")
          .removeClass("fa-spinner fa-spin")
          .addClass("fa-pen");
        this.initializeConfigValues();
        alert(
          "Parameter must be a positive integer. Please use only digits(0-9) for this parameter."
        );
        return false;
      }
    }
    for (var j in floats) {
      if (!/^\+?0*?\.\d+$/.test(floats[j])) {
        $(form)
          .find("i.fa-spinner.fa-spin")
          .removeClass("fa-spinner fa-spin")
          .addClass("fa-pen");
        this.initializeConfigValues();
        alert(
          "Parameter must be a decimal number between 0 and 1. Please use only digits(0-9) and a decimal separator(.) for this parameter."
        );
        return false;
      }
    }
    if (ints["batchSize"] == 0 || ints["cnnEpochs"] == 0) {
      this.initializeConfigValues();
      alert("Parameter must be greater than zero.");
      return false;
    }

    this.update_config = Object.assign(ints, floats);

    this.update_timeout = setTimeout(() => {
      $.ajax({
        url: "/api/update-config",
        method: "POST",
        contentType: "application/json",
        data: JSON.stringify(this.update_config),
        success: (data) => {
          for (var key in this.update_config)
            this[key] = this.update_config[key];

          $(form)
            .find("i.fa-spinner.fa-spin")
            .removeClass("fa-spinner fa-spin")
            .addClass("fa-check");

          setTimeout(function () {
            $(form)
              .find("i.fa-check")
              .removeClass("fa-check")
              .addClass("fa-pen");
          }, 1000);
        },
      }).fail(() => {
        $(form)
          .find("i")
          .removeClass("fa-spinner fa-spin fa-check")
          .addClass("fa-pen");
        this.initializeConfigValues();
        this.clearOutput();
        this.checkConnection();
      });
    }, 400);

    return true;
  }

  checkConnection() {
    const error =
      "<b>Please make sure the server is running and check its console for further information.</b>";
    this.displayAlert(error, "danger");
  }
}

$(() => {
  var main = new Main();

  $("#modeDraw").click((e) => {
    main.useModeDraw(e.currentTarget);
  });

  $("#modeCamera").click((e) => {
    main.useModeCamera(e.currentTarget);
  });

  $("#takePhoto").click((e) => {
    main.onMouseUp();
  });

  $("#clear").click(() => {
    main.initialize();
  });

  $("#classify").click(() => {
    main.recogniseInput();
  });

  $("#trainingDataLabel")[0].addEventListener("keyup", (e) => {
    if (e.keyCode === 13)
      // Enter pressed
      $("#addTrainingData").click();
    return false;
  });

  $("#addTrainingData").click((e) => {
    main.addTrainingData(e.currentTarget, $("#trainingDataLabel").val());
  });

  $("#trainModels").click((e) => {
    main.trainModels(e.currentTarget);
  });

  $("#config-form").submit((e) => {
    if (!main.updateConfig(e.currentTarget))
      $(e.currentTarget)
        .find("i")
        .removeClass("fa-spinner fa-spin fa-check")
        .addClass("fa-pen");
    return false;
  });

  $("#config-form input").each(function () {
    $(this).change((e) => {
      $(this)
        .siblings(".input-group-append")
        .find("i")
        .removeClass("fa-pen fa-check")
        .addClass("fa-spinner fa-spin");

      $("#config-form").submit();
    });
  });

  $("#error .close").click(() => {
    main.dismissAlert();
  });
});
