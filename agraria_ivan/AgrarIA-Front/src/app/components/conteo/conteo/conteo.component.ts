import { Component } from '@angular/core';

@Component({
  selector: 'conteo-upload',
  templateUrl: './conteo.component.html',
  styleUrls: ['./conteo.component.scss']
})
export class ConteoComponent {
  images: string[] = [];
  fileInputs: { [key: string]: File } = {};


  onFileChange(event: Event, input: string): void {
    const inputElement = event.target as HTMLInputElement;
    if (inputElement.files && inputElement.files[0]) {
      const file = inputElement.files[0];
      const reader = new FileReader();
      reader.onload = () => {
        if (input === 'input1') {
          this.images[0] = reader.result as string;
          this.images[2] = reader.result as string;
        } else if (input === 'input2') {
          this.images[1] = reader.result as string;
          this.images[3] = reader.result as string;
        }
      };
      reader.readAsDataURL(file);
    }
  }

  viewImage(inputId: string): void {
    const file = this.fileInputs[inputId];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e: any) => {
        const imageUrl = e.target.result;
        const imgWindow = window.open(imageUrl, '_blank');
        imgWindow?.document.write(`<img src="${imageUrl}" alt="Uploaded Image">`);
      };
      reader.readAsDataURL(file);
    } else {
      console.log(`No file uploaded for ${inputId}`);
    }
  }
  
}
