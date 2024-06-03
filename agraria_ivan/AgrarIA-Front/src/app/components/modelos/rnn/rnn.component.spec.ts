import { ComponentFixture, TestBed } from '@angular/core/testing';

import { RnnComponent } from './rnn.component';

describe('RnnComponent', () => {
  let component: RnnComponent;
  let fixture: ComponentFixture<RnnComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [RnnComponent]
    });
    fixture = TestBed.createComponent(RnnComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
