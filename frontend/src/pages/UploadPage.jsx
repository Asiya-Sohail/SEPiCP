import React, { useState, useCallback, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  FileSpreadsheet,
  CloudUpload,
  X,
  Info,
  Settings2,
  ArrowRight,
  RefreshCw,
  History,
  Loader2,
  CheckCircle2,
  AlertCircle
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import DataPreviewTable from '../components/DataPreviewTable';
import { cn } from '../lib/utils';
import { theme } from '../styles/theme';
import { Button } from '../components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../components/ui/card';
import { Switch } from '../components/ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../components/ui/select';
import { Badge } from '../components/ui/badge';

import { useAuth } from '../context/AuthContext';

const UploadPage = () => {
  const { user, logActivity } = useAuth();
  const [file, setFile] = useState(null);
  const [isCleaning, setIsCleaning] = useState(true);
  const [isUploading, setIsUploading] = useState(false);
  const [previewData, setPreviewData] = useState([]);
  const [dynamicColumns, setDynamicColumns] = useState([]);
  const [isLoadingRecent, setIsLoadingRecent] = useState(false);
  const [overlayStatus, setOverlayStatus] = useState(null);

  const showOverlay = (status, title, message) => {
    setOverlayStatus({ status, title, message });
    if (status !== 'loading') {
      setTimeout(() => {
        setOverlayStatus(null);
        setIsUploading(false);
        setIsLoadingRecent(false);
      }, 3000);
    }
  };

  useEffect(() => {
    const specificId = localStorage.getItem('load_dataset_id');
    if (specificId) {
      handleLoadSpecific(specificId);
      localStorage.removeItem('load_dataset_id');
    }
  }, []);

  const handleLoadSpecific = async (datasetId) => {
    setIsLoadingRecent(true);
    showOverlay('loading', 'Loading Registry', 'Securely pulling your clean historical data from the server vault...');
    const token = localStorage.getItem('research_token');

    try {
      const response = await fetch(`http://127.0.0.1:8000/api/datasets/${datasetId}/`, {
        method: "GET",
        headers: { "Authorization": `Bearer ${token}` }
      });

      if (response.ok) {
        const data = await response.json();
        setPreviewData(data.cleaned_data);
        if (data.cleaned_data && data.cleaned_data.length > 0) {
          const rawCols = Object.keys(data.cleaned_data[0]);
          const cols = rawCols.map(col => ({
            header: col.toUpperCase().replace(/_/g, ' '),
            id: col,
            accessorFn: (row) => row[col],
            cell: (info) => <span className="text-slate-600">{String(info.getValue() || '')}</span>
          }));
          setDynamicColumns(cols);
        }
        showOverlay('success', 'Dataset Loaded', 'Dataset loaded from history successfully!');
      } else {
        showOverlay('error', 'Fetch Failed', 'Failed to fetch the requested dataset.');
      }
    } catch (err) {
      showOverlay('error', 'Network Error', 'Network error loading history dataset.');
    }
  };

  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      setFile(acceptedFiles[0]);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.ms-excel': ['.xls']
    },
    multiple: false
  });



  const handleLoadRecent = async () => {
    setIsLoadingRecent(true);
    showOverlay('loading', 'Loading Registry', 'Securely pulling your clean historical data from the server vault...');
    const token = localStorage.getItem('research_token');

    try {
      const response = await fetch("http://127.0.0.1:8000/api/datasets/", {
        method: "GET",
        headers: {
          "Authorization": `Bearer ${token}`
        }
      });

      if (response.status === 401) {
        showOverlay('error', 'Session Expired', 'Your security token has expired. Please log out and log back in.');
        return;
      }

      const data = await response.json();
      if (response.ok && data.length > 0) {
        const latestDataset = data[0]; // Gets most recent based on django order_by('-created_at')
        setPreviewData(latestDataset.cleaned_data);

        if (latestDataset.cleaned_data && latestDataset.cleaned_data.length > 0) {
          const rawCols = Object.keys(latestDataset.cleaned_data[0]);
          const cols = rawCols.map(col => ({
            header: col.toUpperCase().replace(/_/g, ' '),
            id: col,
            accessorFn: (row) => row[col],
            cell: (info) => <span className="text-slate-600">{String(info.getValue() || '')}</span>
          }));
          setDynamicColumns(cols);
        }
        showOverlay('success', 'Registry Loaded', 'Successfully loaded your most recent historic dataset from the database!');
      } else if (response.ok) {
        showOverlay('error', 'No History', 'No historic datasets found in the database. Please upload a new dataset.');
      } else {
        showOverlay('error', 'Fetch Failed', 'Failed to fetch dataset history.');
      }
    } catch (err) {
      showOverlay('error', 'Server Error', 'Server error while fetching historic data.');
    }
  };

  const handleProcess = async () => {
    if (!file) return;

    setIsUploading(true);
    showOverlay('loading', 'Cleaning Dataset', 'Applying statistical handlers and removing invalid dimensions. This may take a moment...');
    const formData = new FormData();
    formData.append('file', file);

    const token = localStorage.getItem('research_token');

    try {
      const response = await fetch("http://127.0.0.1:8000/api/datasets/upload/", {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${token}`
        },
        body: formData
      });

      if (response.status === 401) {
        showOverlay('error', 'Session Expired', 'Your security token has expired. Please log out and log back in to get a fresh token.');
        return;
      }

      const data = await response.json();

      if (response.ok) {
        setPreviewData(data.cleaned_data);
        const cols = data.columns.map(col => ({
          header: col.toUpperCase().replace(/_/g, ' '),
          id: col, // use `id` plus `accessorFn` to perfectly escape keys with dots like "act._part_1s"
          accessorFn: (row) => row[col],
          cell: (info) => <span className="text-slate-600">{String(info.getValue() || '')}</span>
        }));
        setDynamicColumns(cols);
        logActivity('upload', `Uploaded and processed dataset: ${file.name}`);
        showOverlay('success', 'Processing Complete', 'Data cleaned and processed successfully!');
      } else {
        showOverlay('error', 'Processing Failed', data.details || data.error || 'Unknown error');
      }
    } catch (err) {
      showOverlay('error', 'Server Error', 'Server error connecting to backend.');
    }
  };

  const initialColumns = [
    { header: 'ID', accessorKey: 'id' },
    { header: 'NAME', accessorKey: 'name' },
    { header: 'DEPARTMENT', accessorKey: 'dept' }
  ];

  return (
    <div className="space-y-10 animate-in fade-in slide-in-from-bottom-4 duration-700 pb-20 relative">

      {/* Modern Loading Overlay */}
      <AnimatePresence>
        {overlayStatus && (
          <motion.div
            initial={{ opacity: 0, backdropFilter: 'blur(0px)' }}
            animate={{ opacity: 1, backdropFilter: 'blur(8px)' }}
            exit={{ opacity: 0, backdropFilter: 'blur(0px)' }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40"
          >
            <motion.div
              initial={{ scale: 0.9, y: 20, opacity: 0 }}
              animate={{ scale: 1, y: 0, opacity: 1 }}
              exit={{ scale: 0.9, y: 20, opacity: 0 }}
              transition={{ type: 'spring', damping: 25, stiffness: 300 }}
              className="bg-white rounded-[32px] shadow-2xl p-10 max-w-sm w-full mx-4 flex flex-col items-center text-center relative overflow-hidden ring-1 ring-slate-900/5"
            >
              {/* Spinning gradient background effect */}
              <div className="absolute inset-0 bg-linear-to-tr from-blue-50 to-indigo-50 opacity-50"></div>

              <div className="relative z-10 space-y-6 flex flex-col items-center">
                <div className={cn(
                  "w-20 h-20 rounded-2xl bg-white shadow-lg flex items-center justify-center border",
                  overlayStatus.status === 'error' ? 'border-red-100 shadow-red-100' : 'border-slate-100'
                )}>
                  {overlayStatus.status === 'loading' && <Loader2 className="w-10 h-10 text-[#1e3a8a] animate-spin" />}
                  {overlayStatus.status === 'success' && <CheckCircle2 className="w-10 h-10 text-emerald-500" />}
                  {overlayStatus.status === 'error' && <AlertCircle className="w-10 h-10 text-red-500" />}
                </div>

                <div className="space-y-2">
                  <h3 className="text-xl font-black text-slate-900 tracking-tight">
                    {overlayStatus.title}
                  </h3>
                  <p className="text-sm font-bold text-slate-500 max-w-[250px] leading-relaxed mx-auto">
                    {overlayStatus.message}
                  </p>
                </div>
              </div>

              {/* Fake progress indicator */}
              {overlayStatus.status === 'loading' && (
                <div className="w-full h-1.5 bg-slate-100 rounded-full mt-8 overflow-hidden relative z-10">
                  <motion.div
                    className="absolute inset-y-0 left-0 bg-[#1e3a8a] rounded-full"
                    initial={{ width: "0%" }}
                    animate={{ width: "100%" }}
                    transition={{ duration: 15, ease: "easeOut" }}
                  />
                </div>
              )}
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl sm:text-3xl font-black tracking-tight text-slate-900">Upload Research Data</h1>
          <p className="text-slate-500 mt-1.5 text-xs sm:text-sm font-medium">Manage and preprocess your .xlsx datasets.</p>
        </div>
        <Button
          variant="outline"
          disabled={isLoadingRecent}
          onClick={handleLoadRecent}
          className="h-10 px-4 rounded-xl font-bold border-slate-200 bg-white text-slate-600 shadow-sm flex items-center gap-2 text-xs"
        >
          {isLoadingRecent ? <RefreshCw className="w-4 h-4 animate-spin text-slate-400" /> : <History className="w-4 h-4 text-slate-400" />}
          {isLoadingRecent ? 'Loading...' : 'Recent Uploads'}
        </Button>
      </div>

      <div className="grid lg:grid-cols-12 gap-6 lg:gap-8">
        <div className="lg:col-span-8 space-y-6">
          {/* Main Upload Area */}
          <div className="bg-white rounded-[24px] border border-slate-200 shadow-xl shadow-slate-200/30 p-1 bg-linear-to-b from-white to-slate-50/30">
            <div
              {...getRootProps()}
              className={cn(
                "rounded-[20px] border-2 border-dashed transition-all py-10 sm:py-14 flex flex-col items-center justify-center text-center gap-4 group cursor-pointer relative overflow-hidden",
                isDragActive ? "bg-blue-50/50 border-[#1e3a8a]" : "border-slate-200 hover:border-[#1e3a8a]/40 bg-white"
              )}
            >
              <div className="w-12 h-12 sm:w-14 sm:h-14 rounded-full bg-blue-50 flex items-center justify-center text-[#1e3a8a] shadow-sm transform group-hover:scale-110 transition-all">
                <CloudUpload className="w-6 h-6" />
              </div>
              <div className="space-y-1 px-4">
                <p className="text-lg font-black text-slate-900">Click to upload or drag and drop</p>
                <p className="text-xs font-bold text-slate-400">Excel files (.xlsx) only. Maximum file size 50MB.</p>
              </div>

              {file && (
                <div className="mt-2 flex items-center gap-3 px-4 py-2 bg-white rounded-lg border border-slate-100 shadow-md animate-in zoom-in-95 duration-300 mx-4">
                  <FileSpreadsheet className="w-4 h-4 text-[#1e3a8a]" />
                  <span className="text-[12px] font-black text-slate-700 truncate max-w-[150px]">{file.name}</span>
                  <Badge className="bg-emerald-50 text-emerald-600 border-emerald-100 font-black text-[9px] px-2 py-0.5">Ready</Badge>
                  <button onClick={(e) => { e.stopPropagation(); setFile(null); }} className="p-1 hover:bg-slate-100 rounded-md text-slate-400 transition-all">
                    <X className="w-3.5 h-3.5" />
                  </button>
                </div>
              )}

              {isDragActive && (
                <div className="absolute inset-0 bg-blue-500/5 backdrop-blur-[1px] flex items-center justify-center">
                  <p className="text-[#1e3a8a] font-black text-xs sm:text-sm uppercase tracking-widest bg-white/95 px-6 py-3 rounded-xl shadow-2xl border border-blue-100">Drop Registry Here</p>
                </div>
              )}
              <input {...getInputProps()} />
            </div>
          </div>

        </div>

        <div className="lg:col-span-4 space-y-6">
          {/* File Details Card */}
          <Card className="rounded-[24px] border-slate-200 shadow-lg shadow-slate-200/20 bg-white p-6">
            <h3 className="font-black text-slate-900 mb-6 text-[11px] uppercase tracking-widest leading-none">File Details</h3>
            <div className="space-y-4">
              {[
                { label: 'File Name', value: file?.name || (previewData.length > 0 ? 'Loaded Registry' : 'No file selected') },
                { label: 'Size', value: file ? (file.size / (1024 * 1024)).toFixed(1) + ' MB' : (previewData.length > 0 ? 'N/A' : '0 MB') },
                { label: 'Rows', value: previewData.length > 0 ? previewData.length.toLocaleString() : '0' },
                { label: 'Columns', value: dynamicColumns.length > 0 ? dynamicColumns.length : '0' },
              ].map((detail) => (
                <div key={detail.label} className="flex justify-between items-center text-[12px]">
                  <span className="text-slate-400 font-bold">{detail.label}</span>
                  <span className="font-black text-slate-900 truncate max-w-[120px]">{detail.value}</span>
                </div>
              ))}
            </div>

            <div className="mt-8 p-4 bg-blue-50 rounded-xl flex gap-3 items-start border border-blue-100">
              <Info className="w-4 h-4 shrink-0 text-[#1e3a8a] mt-0.5" />
              <p className="text-[10px] text-[#1e3a8a] leading-relaxed font-black">
                Verify column headers below before confirming upload.
              </p>
            </div>
          </Card>

          <div className="flex flex-col sm:flex-row lg:flex-col gap-3">
            <Button
              disabled={!file || isUploading}
              onClick={handleProcess}
              className="flex-1 h-12 rounded-xl bg-[#1e3a8a] text-white font-black text-[12px] uppercase tracking-widest shadow-lg shadow-blue-900/20 hover:bg-[#1a337a] transition-all flex items-center justify-center gap-2"
            >
              {isUploading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  Processing...
                </>
              ) : (
                <>
                  Confirm & Process
                  <ArrowRight className="w-4 h-4" />
                </>
              )}
            </Button>
            <Button
              variant="outline"
              onClick={() => setFile(null)}
              className="flex-1 h-12 rounded-xl border-slate-200 bg-white text-slate-500 font-bold text-[12px] hover:bg-slate-50 transition-all border-2"
            >
              Cancel
            </Button>
          </div>
        </div>
      </div>

      {/* Data Preview Section */}
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h2 className="text-2xl font-black text-slate-900 tracking-tight">Data Preview</h2>
          <div className="flex items-center gap-4">
            <button className="p-2.5 rounded-xl hover:bg-white text-slate-400 hover:text-slate-900 transition-all border border-transparent hover:border-slate-200">
              <RefreshCw className="w-5 h-5" />
            </button>
            <button className="p-2.5 rounded-xl hover:bg-white text-slate-400 hover:text-slate-900 transition-all border border-transparent hover:border-slate-200">
              <ArrowRight className="w-5 h-5" />
            </button>
          </div>
        </div>
        <div className="bg-white rounded-[32px] border border-slate-200 shadow-2xl shadow-slate-200/30 overflow-hidden">
          <DataPreviewTable
            data={previewData.length > 0 ? previewData : []}
            columns={dynamicColumns.length > 0 ? dynamicColumns : initialColumns}
          />
        </div>
      </div>
    </div>
  );
};

export default UploadPage;

