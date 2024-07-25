import { Component } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { DiseaseInfo } from './model/disease-info.model';  // Importa la interfaz
import { DISEASES_INFO } from './model/diseases';  // Importa el objeto con los datos de las enfermedades

// Agrega esta declaración en un archivo de declaraciones globales o al principio de tu archivo TypeScript
declare var bootstrap: any;


@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.scss']
})
export class HomeComponent {
  imagePreview: string | ArrayBuffer | null = null;
  predictionResult: DiseaseInfo | null = null;  // Cambiado a DiseaseInfo
  selectedFile: File | null = null;

  constructor(private http: HttpClient) { }

  onFileSelected(event: Event): void {
    const file = (event.target as HTMLInputElement).files?.[0];
    if (file) {
      this.selectedFile = file;
      const reader = new FileReader();
      reader.onload = () => {
        this.imagePreview = reader.result;
      };
      reader.readAsDataURL(file);
    }
  }

  predict(): void {
    if (this.selectedFile) {
      const formData = new FormData();
      formData.append('file', this.selectedFile, this.selectedFile.name);
  
      this.http.post<any>('http://localhost:5000/predict', formData).subscribe(
        response => {
          // Obtenemos la clase predicha y la confianza desde la respuesta
          const predictedClass = response.class;
          const confidence = response.confidence;
  
          // Buscamos la información de la enfermedad en DISEASES_INFO
          this.predictionResult = DISEASES_INFO[predictedClass] || null;
  
          if (this.predictionResult) {
            // Añadimos la confianza a los resultados para mostrar en el modal
            this.predictionResult.confidence = confidence;
            this.showPredictionModal();
          } else {
            alert(`No se encontró información para la clase predicha: ${predictedClass}`);
          }
        },
        (error: HttpErrorResponse) => {
          console.error('Error:', error);
          alert('Error realizando la predicción: ' + error.message + '. ' + (error.error?.error || ''));
        }
      );
    } else {
      alert('No hay imagen seleccionada.');
    }
  }
  

  private showPredictionModal(): void {
    const modalElement = document.getElementById('predictionModal');
    if (modalElement) {
      const modal = new bootstrap.Modal(modalElement);
      modal.show();
    }
  }

  refreshPage(): void {
    window.location.reload();
  }

  

}
