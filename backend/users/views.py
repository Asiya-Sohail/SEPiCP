# from django.shortcuts import render
# from rest_framework.decorators import api_view
from rest_framework import generics, permissions
from rest_framework.response import Response
from rest_framework.views import APIView
from django.contrib.auth import get_user_model
from rest_framework import status
from .models import Dataset
from .serializers import RegisterSerializer, UserSerializer, DatasetSerializer      
from rest_framework.permissions import IsAdminUser, IsAuthenticated
from .services.cleaning_service import clean_dataset

User = get_user_model()


#CreateAPIView internally does 
# serializer = RegisterSerializer(data=request.data)
# if serializer.is_valid():
#     serializer.save()
# else:
#     return serializer.errors

# 1️⃣ Register View
class RegisterView(generics.CreateAPIView):
    queryset = User.objects.all()
    serializer_class = RegisterSerializer


# 2️⃣ Get Logged-in User Profile
class ProfileView(generics.RetrieveAPIView):
    serializer_class = UserSerializer
    permission_classes = [IsAuthenticated]

    def get_object(self):
        return self.request.user


# 3️⃣ Admin - View All Users
class AdminUserListView(generics.ListAPIView):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [IsAdminUser]


# 4️⃣ Admin Dashboard Stats (Optional but Professional)
class AdminDashboardView(APIView):
    permission_classes = [IsAdminUser]

    def get(self, request):
        total_users = User.objects.count()
        admin_users = User.objects.filter(is_staff=True).count()

        return Response({
            "total_users": total_users,
            "admin_users": admin_users,
        })

class UserDatasetListView(generics.ListAPIView):
    serializer_class = DatasetSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Dataset.objects.filter(user=self.request.user).order_by('-created_at')

from rest_framework.parsers import MultiPartParser, FormParser
import pandas as pd
import json

class UploadDatasetView(APIView):
    permission_classes = [IsAuthenticated]
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        file_obj = request.data.get('file')
        
        if not file_obj:
            return Response(
                {"error": "No file provided"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # Calculate file hash to prevent duplicates
            import hashlib
            file_content = file_obj.read()
            file_hash = hashlib.sha256(file_content).hexdigest()
            file_obj.seek(0) # Reset pointer so pandas can read it
            
            existing_dataset = Dataset.objects.filter(user=request.user, file_hash=file_hash).first()
            if existing_dataset:
                cleaned_columns = list(existing_dataset.cleaned_data[0].keys()) if existing_dataset.cleaned_data else []
                return Response({
                    "id": existing_dataset.id,
                    "cleaned_data": existing_dataset.cleaned_data[:100],
                    "columns": cleaned_columns,
                    "message": "Dataset already exists. Loaded from history."
                }, status=status.HTTP_200_OK)

            # Read Excel file
            if file_obj.name.endswith('.xlsx') or file_obj.name.endswith('.xls'):
                df = pd.read_excel(file_obj)
            else:
                return Response(
                    {"error": "Unsupported file format. Please upload .xlsx or .xls"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            import json
            # Pandas to_json automatically handles all native datetimes, NaNs, and converts them
            # safely to standard pure JSON strings and nulls without crashing.
            original_data_json = df.to_json(orient='records', date_format='iso')
            original_data = json.loads(original_data_json)

            # Clean dataset
            cleaned_data = clean_dataset(original_data)

            # Store in database
            dataset = Dataset.objects.create(
                user=request.user,
                file_hash=file_hash,
                cleaned_data=cleaned_data
            )

            # Return cleaned data for preview
            # Extract keys from cleaned_data to ensure mapping works
            cleaned_columns = list(cleaned_data[0].keys()) if cleaned_data else []

            return Response({
                "id": dataset.id,
                "cleaned_data": cleaned_data[:100],
                "columns": cleaned_columns
            }, status=status.HTTP_201_CREATED)

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return Response(
                {"error": "Processing failed", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class UserDatasetListView(generics.ListAPIView):
    serializer_class = DatasetSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Dataset.objects.filter(user=self.request.user).order_by('-created_at')

class DatasetDetailView(generics.RetrieveAPIView):
    serializer_class = DatasetSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Dataset.objects.filter(user=self.request.user)