// email.service.ts
import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class EmailService {
  private apiUrl = 'https://agrovisionai-0a757d03ae4c.herokuapp.com/send-email'; // URL de tu API backend

  constructor(private http: HttpClient) { }

  sendEmail(data: any): Observable<any> {
    return this.http.post(this.apiUrl, data);
  }
}
