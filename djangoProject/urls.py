from django.contrib import admin
from django.urls import include, path
from dressapp.views import upload_image
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('admin/', admin.site.urls),
    path('dressapp/', include('dressapp.urls')),
    path('', upload_image, name='home'),  # Route the root URL
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

