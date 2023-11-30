import numpy as np
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .utils import extract_img_features, recommendd, model, img_files_list, features_list


def upload_image(request):
    context = {}
    if request.method == 'POST' and request.FILES.get('image'):
        myfile = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_path = fs.path(filename)

        features = extract_img_features(uploaded_file_path, model)
        if features is None or np.isnan(features).any():
            context['error'] = "Error in processing image or extracting features."
            return render(request, 'upload.html', context)

        features_reshaped = np.array(features).reshape(1, -1)
        img_indices = recommendd(features_reshaped, features_list)
        recommended_images = [img_files_list[idx] for idx in img_indices[0]]

        context['uploaded_file_url'] = fs.url(filename)
        context['recommendations'] = recommended_images

    return render(request, 'upload.html', context)
